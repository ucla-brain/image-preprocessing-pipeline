/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput TIFF Z-slice saver for 3D MATLAB volumes.
  Each slice is saved to a separate TIFF file using multithreading with NUMA
  and hugepage support for high throughput and low latency.

  USAGE:
      save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])

  INPUT:
    volume      : 3D uint8/uint16 MATLAB array in [X Y Z] or [Y X Z] layout.
    fileList    : 1xZ cell array of filenames (one per Z-slice).
    isXYZ       : boolean or numeric, true if data is [X Y Z] layout.
    compression : string, one of: 'none', 'lzw', or 'deflate'.
    nThreads    : (optional) number of threads to use (default = physical cores)

  FEATURES:
    • Atomic task dispatch across multiple threads.
    • NUMA-aware memory pinning and optional hugepage use.
    • Direct write for uncompressed [X Y Z], otherwise transpose/copy.
    • Safe rename from .tmp → final slice file.
    • RAII for all file handles and allocations.
    • Cross-platform (Linux and Windows).
    • Thread-safe error collection and reporting.

  LIMITATIONS:
    • Grayscale slices only (1 channel).
    • No built-in retry on I/O error.
    • Assumes local filesystem for max performance.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iostream>
#include <sstream>

#if defined(__linux__)
  #include <errno.h>
  #include <fcntl.h>
  #include <numa.h>
  #include <numaif.h>
  #include <sched.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #define ACCESS access
#else
  #include <windows.h>
  #include <io.h>
  #define ACCESS _access
#endif

namespace detail {

/* Check for NUMA interleaving via /proc/self/status */
inline bool numa_interleaving_forced() {
#if defined(__linux__)
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return false;
    char buf[256];
    while (fgets(buf, sizeof buf, f)) {
        if (strncmp(buf, "Mems_allowed:", 13) == 0) {
            std::string s(buf + 13);
            int bits = std::count(s.begin(), s.end(), '1');
            fclose(f);
            return bits > 1;
        }
    }
    fclose(f);
#endif
    return false;
}

/* Optional thread affinity to specific NUMA node */
inline void pin_thread_to_numa_node(int node, bool disable_pinning) {
#if defined(__linux__)
    if (disable_pinning || numa_available() == -1 || node < 0) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    bitmask* cpumask = numa_allocate_cpumask();
    if (numa_node_to_cpus(node, cpumask) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (numa_bitmask_isbitset(cpumask, i))
                CPU_SET(i, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
    numa_free_cpumask(cpumask);
#endif
}

/* Ensure we can overwrite the output file */
inline void ensure_writable(const std::string& path) {
    if (ACCESS(path.c_str(), F_OK) == 0 && ACCESS(path.c_str(), W_OK) != 0)
        throw std::runtime_error("Cannot overwrite read-only file: " + path);
}

/* Per-thread scratch buffer with NUMA-aware allocation */
struct ScratchBuffer {
    uint8_t* ptr = nullptr;
    size_t   size = 0;
    enum { NONE, MMAP, NUMA, MALLOC } allocMode = NONE;
    int numaNode = -1;

    void ensure(size_t bytes, int node, bool disable_numa) {
        if (ptr && size >= bytes) return;
        release();

#if defined(__linux__)
        if (!disable_numa && numa_available() != -1) {
            void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (mem != MAP_FAILED) {
                unsigned long mask = 1UL << node;
                if (mbind(mem, bytes, MPOL_BIND, &mask, sizeof(mask) * 8, 0) == 0) {
                    ptr = static_cast<uint8_t*>(mem);
                    size = bytes;
                    allocMode = MMAP;
                    numaNode = node;
                    return;
                } else {
                    munmap(mem, bytes);
                }
            }
            ptr = static_cast<uint8_t*>(numa_alloc_onnode(bytes, node));
            if (ptr) {
                size = bytes;
                allocMode = NUMA;
                numaNode = node;
                return;
            }
        }
#endif
        ptr = static_cast<uint8_t*>(malloc(bytes));
        if (!ptr) throw std::bad_alloc();
        size = bytes;
        allocMode = MALLOC;
        numaNode = -1;
    }

    void release() {
        if (!ptr) return;
#if defined(__linux__)
        if (allocMode == MMAP) munmap(ptr, size);
        else if (allocMode == NUMA) numa_free(ptr, size);
        else free(ptr);
#else
        free(ptr);
#endif
        ptr = nullptr;
        size = 0;
        allocMode = NONE;
        numaNode = -1;
    }

    ~ScratchBuffer() { release(); }
};

static thread_local ScratchBuffer threadScratch;

/* TIFF RAII wrapper */
struct TiffHandle {
    TIFF* tif = nullptr;
    TiffHandle(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif) throw std::runtime_error("Cannot open TIFF: " + path);
    }
    ~TiffHandle() { if (tif) TIFFClose(tif); }
    TIFF* get() const { return tif; }
};

} // namespace detail

struct SliceJob {
    const uint8_t* base = nullptr;
    size_t offset = 0;
    mwSize rows = 0, cols = 0;
    bool isXYZ = true;
    mxClassID classId = mxUINT8_CLASS;
    uint16_t compressionTag = COMPRESSION_NONE;
    size_t bytesPerPx = 1, bytesPerSlice = 0;
    size_t zIndex = 0;
    std::string filename;
};

/* Save a single Z-slice to disk */
static void save_slice(const SliceJob& job, int numaNode, bool disable_numa) {
    using namespace detail;
    ensure_writable(job.filename);
    const uint8_t* src = job.base + job.offset;
    mwSize srcRows = job.isXYZ ? job.cols : job.rows;
    mwSize srcCols = job.isXYZ ? job.rows : job.cols;

    std::string tmpPath = job.filename + ".tmp";
    TiffHandle tif(tmpPath, "w");
    TIFF* tf = tif.get();

    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, srcCols);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, (job.bytesPerPx == 2) ? 16 : 8);
    TIFFSetField(tf, TIFFTAG_COMPRESSION, job.compressionTag);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, srcRows);

    const uint8_t* buf = nullptr;

    if (job.isXYZ && job.compressionTag == COMPRESSION_NONE) {
        buf = src;
    } else {
        threadScratch.ensure(job.bytesPerSlice, numaNode, disable_numa);
        uint8_t* dst = threadScratch.ptr;

        if (!job.isXYZ) {
            for (mwSize c = 0; c < srcCols; ++c) {
                const uint8_t* srcCol = src + c * job.rows * job.bytesPerPx;
                for (mwSize r = 0; r < srcRows; ++r) {
                    size_t idx = (r * srcCols + c) * job.bytesPerPx;
                    std::memcpy(dst + idx, srcCol + r * job.bytesPerPx, job.bytesPerPx);
                }
            }
        } else {
            std::memcpy(dst, src, job.bytesPerSlice);
        }
        buf = dst;
    }

    tsize_t written = (job.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tf, 0, const_cast<uint8_t*>(buf), job.bytesPerSlice)
        : TIFFWriteEncodedStrip(tf, 0, const_cast<uint8_t*>(buf), job.bytesPerSlice);

    if (written < 0)
        throw std::runtime_error("TIFF write failed on slice " + std::to_string(job.zIndex));

#if defined(__linux__)
    unlink(job.filename.c_str());
#else
    _unlink(job.filename.c_str());
#endif
    if (rename(tmpPath.c_str(), job.filename.c_str()) != 0)
        throw std::runtime_error("Rename failed for slice " + std::to_string(job.zIndex));
}

/* Job queue with atomic dispatch */
struct JobQueue {
    const std::vector<SliceJob>* jobs;
    std::atomic_size_t next{0};
    std::mutex errLock;
    std::vector<std::string> errors;
};

/* Thread worker */
static void worker(JobQueue& q, int numaNode, bool disable_pinning, bool disable_numa) {
    detail::pin_thread_to_numa_node(numaNode, disable_pinning);
    constexpr size_t chunkSize = 8;
    while (true) {
        size_t start = q.next.fetch_add(chunkSize);
        if (start >= q.jobs->size()) break;
        size_t end = std::min(start + chunkSize, q.jobs->size());

        for (size_t i = start; i < end; ++i) {
            try {
                save_slice((*q.jobs)[i], numaNode, disable_numa);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> g(q.errLock);
                q.errors.push_back(e.what());
            }
        }
    }
}

/* Physical core counter */
size_t count_physical_cores() {
#if defined(__linux__)
    std::vector<std::pair<int, int>> pairs;
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return std::thread::hardware_concurrency();
    char line[128];
    int p = -1, c = -1;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "physical id\t: %d", &p) == 1) continue;
        if (sscanf(line, "core id\t: %d", &c) == 1)
            pairs.emplace_back(p, c);
    }
    fclose(f);
    std::sort(pairs.begin(), pairs.end());
    auto last = std::unique(pairs.begin(), pairs.end());
    return static_cast<size_t>(std::distance(pairs.begin(), last));
#else
    return std::thread::hardware_concurrency();
#endif
}

/* MEX entry point */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Usage: save_bl_tif(vol, list, isXYZ, comp[, threads])");

        const mxArray* volMx = prhs[0];
        if (!mxIsUint8(volMx) && !mxIsUint16(volMx))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8 or uint16");

        const mwSize* dims = mxGetDimensions(volMx);
        mwSize rows = dims[0], cols = dims[1], zs = (mxGetNumberOfDimensions(volMx) == 3) ? dims[2] : 1;
        bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]));

        char* compC = mxArrayToUTF8String(prhs[3]);
        std::string compStr(compC); mxFree(compC);

        uint16_t compTag = (compStr == "lzw")     ? COMPRESSION_LZW
                         : (compStr == "deflate") ? COMPRESSION_DEFLATE
                         : (compStr == "none")    ? COMPRESSION_NONE
                         : throw std::runtime_error("Invalid compression type");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != zs)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must match #Z slices");

        std::vector<std::string> fileList(zs);
        for (mwSize k = 0; k < zs; ++k) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], k));
            fileList[k] = s; mxFree(s);
        }

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(volMx));
        mxClassID id = mxGetClassID(volMx);
        size_t bpp = (id == mxUINT16_CLASS) ? 2 : 1;
        size_t bpslice = rows * cols * bpp;

        std::vector<SliceJob> jobs(zs);
        for (mwSize z = 0; z < zs; ++z)
            jobs[z] = {base, z * bpslice, rows, cols, isXYZ, id, compTag, bpp, bpslice, z, fileList[z]};

        size_t nThreads = (nrhs == 5) ? std::min<size_t>(mxGetScalar(prhs[4]), zs)
                                     : std::min(count_physical_cores(), static_cast<size_t>(zs));
        nThreads = std::max<size_t>(1, nThreads);

        size_t numaNodes = 1;
        bool disable_pinning = false, disable_numa = false;
#if defined(__linux__)
        if (numa_available() != -1) {
            numaNodes = numa_max_node() + 1;
            disable_pinning = detail::numa_interleaving_forced();
            disable_numa = disable_pinning;
        }
#endif

        JobQueue queue{&jobs};
        std::vector<std::thread> threads;
        for (size_t t = 0; t < nThreads; ++t)
            threads.emplace_back(worker, std::ref(queue), static_cast<int>(t % numaNodes), disable_pinning, disable_numa);
        for (auto& th : threads) th.join();

        if (!queue.errors.empty()) {
            std::ostringstream msg;
            msg << "Errors:\n";
            for (const auto& e : queue.errors) msg << "• " << e << '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.str().c_str());
        }

        if (nlhs > 0) plhs[0] = const_cast<mxArray*>(volMx);
    } catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", ex.what());
    }
}
