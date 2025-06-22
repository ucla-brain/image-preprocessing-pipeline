/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput TIFF Z-slice saver for 3D MATLAB volumes. Each slice is saved
  to a separate TIFF file using multithreading and optional compression.

  USAGE:
      save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])

  INPUT:
    volume      : 3D uint8/uint16 MATLAB array in [X Y Z] or [Y X Z] order.
    fileList    : 1xZ cell array of filenames (one per Z-slice).
    isXYZ       : true if data is [X Y Z], false for [Y X Z].
    compression : 'none', 'lzw', or 'deflate'
    nThreads    : (optional) number of threads (defaults to # physical cores)

  FEATURES:
    - Uses per-thread scratch buffers (NUMA-aware, hugepage-backed if available)
    - Chunked atomic dispatch
    - Per-slice error tracking and aggregation
    - Avoids thread creation overhead by eager warmup
    - RAII for resource safety

  LIMITATIONS:
    - Grayscale only (1 channel)
    - Each slice must be writable to disk
    - No TIFF metadata/tags beyond raw pixel encoding

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <cstdlib>

#if defined(__linux__)
  #include <numa.h>
  #include <numaif.h>
  #include <sched.h>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
  #define ACCESS access
#else
  #include <windows.h>
  #include <io.h>
  #define ACCESS _access
#endif

// -----------------------------------------------------------------------------
// NUMA + Hugepage-aware per-thread scratch buffer
// -----------------------------------------------------------------------------
struct ScratchBuffer {
    uint8_t* data = nullptr;
    size_t size = 0;

    void ensure(size_t bytes, int numaNode) {
        if (data && size >= bytes) return;
        release();

#if defined(__linux__)
        void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (mem != MAP_FAILED) {
            unsigned long mask = 1UL << numaNode;
            mbind(mem, bytes, MPOL_BIND, &mask, sizeof(mask) * 8, 0);
            data = static_cast<uint8_t*>(mem);
            size = bytes;
            return;
        }

        if (numa_available() != -1) {
            data = static_cast<uint8_t*>(numa_alloc_onnode(bytes, numaNode));
            size = bytes;
            return;
        }
#endif
        data = static_cast<uint8_t*>(malloc(bytes));
        if (!data) throw std::bad_alloc();
        size = bytes;
    }

    void release() {
#if defined(__linux__)
        if (data) munmap(data, size);
#else
        free(data);
#endif
        data = nullptr;
        size = 0;
    }

    ~ScratchBuffer() { release(); }
};

static thread_local ScratchBuffer tlsBuffer;

// -----------------------------------------------------------------------------
// NUMA-aware thread pinning
// -----------------------------------------------------------------------------
void pin_to_numa_node(int node) {
#if defined(__linux__)
    if (node < 0 || numa_available() == -1) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    bitmask* cpumask = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpumask);
    for (int i = 0; i < CPU_SETSIZE; ++i)
        if (numa_bitmask_isbitset(cpumask, i))
            CPU_SET(i, &cpuset);
    numa_free_cpumask(cpumask);

    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
}

// -----------------------------------------------------------------------------
// TIFF slice writer job
// -----------------------------------------------------------------------------
struct SliceJob {
    const uint8_t* base = nullptr;
    size_t offset = 0;
    mwSize rows = 0, cols = 0;
    bool isXYZ = true;
    mxClassID type = mxUINT8_CLASS;
    uint16_t compression = COMPRESSION_NONE;
    size_t bytesPerPixel = 1;
    size_t sliceBytes = 0;
    size_t zIndex = 0;
    std::string path;
};

// Write one Z-slice to a temporary TIFF and rename it
void write_slice(const SliceJob& job, int numaNode) {
    const uint8_t* src = job.base + job.offset;
    mwSize height = job.isXYZ ? job.cols : job.rows;
    mwSize width  = job.isXYZ ? job.rows : job.cols;

    std::string tmpPath = job.path + ".tmp";
    TIFF* tf = TIFFOpen(tmpPath.c_str(), "w");
    if (!tf) throw std::runtime_error("Cannot open TIFF: " + tmpPath);

    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, (job.bytesPerPixel == 2) ? 16 : 8);
    TIFFSetField(tf, TIFFTAG_COMPRESSION, job.compression);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, height);

    const uint8_t* buffer = nullptr;

    if (job.isXYZ && job.compression == COMPRESSION_NONE) {
        buffer = src;
    } else {
        tlsBuffer.ensure(job.sliceBytes, numaNode);
        uint8_t* dst = tlsBuffer.data;

        if (!job.isXYZ) {
            for (mwSize c = 0; c < width; ++c) {
                const uint8_t* srcCol = src + c * job.rows * job.bytesPerPixel;
                for (mwSize r = 0; r < height; ++r) {
                    size_t dstIdx = (r * width + c) * job.bytesPerPixel;
                    std::memcpy(dst + dstIdx, srcCol + r * job.bytesPerPixel, job.bytesPerPixel);
                }
            }
        } else {
            std::memcpy(dst, src, job.sliceBytes);
        }

        buffer = dst;
    }

    tsize_t written = (job.compression == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tf, 0, const_cast<uint8_t*>(buffer), job.sliceBytes)
        : TIFFWriteEncodedStrip(tf, 0, const_cast<uint8_t*>(buffer), job.sliceBytes);

    TIFFClose(tf);

    if (written < 0)
        throw std::runtime_error("TIFF write failed at Z=" + std::to_string(job.zIndex));

#if defined(__linux__)
    unlink(job.path.c_str());
#else
    _unlink(job.path.c_str());
#endif
    if (rename(tmpPath.c_str(), job.path.c_str()) != 0)
        throw std::runtime_error("Rename failed: " + job.path);
}

// -----------------------------------------------------------------------------
// Atomic dispatch queue
// -----------------------------------------------------------------------------
struct JobQueue {
    const std::vector<SliceJob>* jobs;
    std::atomic_size_t next{0};
    std::mutex errorLock;
    std::vector<std::string> errors;
};

// Thread worker function
void thread_worker(JobQueue& queue, int numaNode) {
    pin_to_numa_node(numaNode);
    constexpr size_t batchSize = 4;

    while (true) {
        size_t start = queue.next.fetch_add(batchSize);
        if (start >= queue.jobs->size()) break;

        size_t end = std::min(start + batchSize, queue.jobs->size());
        for (size_t i = start; i < end; ++i) {
            try {
                write_slice((*queue.jobs)[i], numaNode);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(queue.errorLock);
                queue.errors.push_back(e.what());
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Physical core count (Linux only)
// -----------------------------------------------------------------------------
size_t count_physical_cores() {
#if defined(__linux__)
    std::vector<std::pair<int, int>> seen;
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return std::thread::hardware_concurrency();

    char line[128];
    int phys = -1, core = -1;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "physical id : %d", &phys) == 1) continue;
        if (sscanf(line, "core id : %d", &core) == 1)
            seen.emplace_back(phys, core);
    }
    fclose(f);
    std::sort(seen.begin(), seen.end());
    auto last = std::unique(seen.begin(), seen.end());
    return std::distance(seen.begin(), last);
#else
    return std::thread::hardware_concurrency();
#endif
}

// -----------------------------------------------------------------------------
// Main MEX entry point
// -----------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads])");

        const mxArray* vol = prhs[0];
        const mwSize* dims = mxGetDimensions(vol);
        if (!mxIsUint8(vol) && !mxIsUint16(vol))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8 or uint16");

        mwSize height = dims[0], width = dims[1];
        mwSize depth = (mxGetNumberOfDimensions(vol) == 3) ? dims[2] : 1;
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(vol));
        mxClassID id = mxGetClassID(vol);
        size_t bpp = (id == mxUINT16_CLASS) ? 2 : 1;
        size_t sliceBytes = height * width * bpp;

        bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]));
        std::string compStr = mxArrayToString(prhs[3]);
        uint16_t compTag =
              (compStr == "none")    ? COMPRESSION_NONE
            : (compStr == "lzw")     ? COMPRESSION_LZW
            : (compStr == "deflate") ? COMPRESSION_DEFLATE
            : throw std::runtime_error("Invalid compression type");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != depth)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must match Z-slices");

        std::vector<std::string> fileList(depth);
        for (mwSize i = 0; i < depth; ++i) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
            fileList[i] = s;
            mxFree(s);
        }

        std::vector<SliceJob> jobs(depth);
        for (mwSize z = 0; z < depth; ++z)
            jobs[z] = {base, z * sliceBytes, height, width, isXYZ,
                       id, compTag, bpp, sliceBytes, z, fileList[z]};

        size_t nThreads = (nrhs == 5)
                        ? std::min<size_t>(mxGetScalar(prhs[4]), depth)
                        : std::min(count_physical_cores(), static_cast<size_t>(depth));
        nThreads = std::max<size_t>(1, nThreads);

        size_t numaNodes = 1;
#if defined(__linux__)
        if (numa_available() != -1)
            numaNodes = numa_max_node() + 1;
#endif

        // Preload TIFF before thread spawn to avoid dynamic TLS latency
        TIFF* preload = TIFFOpen("/dev/null", "r");
        if (preload) TIFFClose(preload);

        JobQueue queue{&jobs};
        std::vector<std::thread> pool;
        for (size_t i = 0; i < nThreads; ++i)
            pool.emplace_back(thread_worker, std::ref(queue), static_cast<int>(i % numaNodes));
        for (auto& t : pool) t.join();

        if (!queue.errors.empty()) {
            std::string msg = "Errors:\n";
            for (const std::string& e : queue.errors)
                msg += "• " + e + "\n";
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        if (nlhs > 0) plhs[0] = const_cast<mxArray*>(vol);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
