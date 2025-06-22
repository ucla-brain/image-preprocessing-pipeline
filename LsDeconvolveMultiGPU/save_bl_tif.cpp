/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------

  DESCRIPTION
    High-throughput TIFF saver for 3-D MATLAB volumes (one TIFF per Z-slice).
    NUMA-aware. Robust error reporting. Per-slice parallelization.

  USAGE
      save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])

        volume      : uint8 / uint16 3-D array, [X Y Z] when isXYZ == true,
                       otherwise [Y X Z].
        fileList    : 1×Z cell-array of filenames (one per slice).
        isXYZ       : logical or numeric. true  == data is [X Y Z]
                                           false == data is [Y X Z]
        compression : 'none' | 'lzw' | 'deflate'
        nThreads    : optional, maximum number of worker threads
                      (default = number of physical CPU cores, not logical).

  FEATURES
    • Per-slice TIFF output with optional LZW / Deflate compression.
    • NUMA-aware, per-thread, reusable scratch buffers (huge-page if possible).
    • NUMA-local thread binding for best performance (Linux).
    • Safe overwrite detection (refuses to touch read-only paths).
    • Robust error aggregation & throw-on-first-failure semantics.
    • No memory leaks — correct deallocator chosen for every allocator.
    • Portable (Linux & Windows); Linux build autodetects NUMA topology.

  LIMITATIONS
    • Grayscale only (1 sample / pixel, 8 or 16 bit).
    • Large images × many threads ⇒ substantial scratch-buffer RAM.
    • Assumes local filesystem — no special handling for network mounts.

  AUTHOR
    Keivan Moradi  (refactored with ChatGPT-o3 assistance)

  LICENSE
    GNU GPL v3 <https://www.gnu.org/licenses/>
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

#if defined(__linux__)
  #include <errno.h>
  #include <fcntl.h>
  #include <numa.h>
  #include <numaif.h>
  #include <sched.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <sys/uio.h>
  #include <unistd.h>
  #define ACCESS access
#elif defined(_WIN32)
  #include <windows.h>
  #include <io.h>
  #define ACCESS _access
#endif

//==============================================================================
//  NUMA Thread Pinning (Linux)
//==============================================================================
#if defined(__linux__)
void pin_thread_to_numa_node(int node) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    // Bind to any CPU on the NUMA node
    struct bitmask *node_cpus = numa_allocate_cpumask();
    numa_node_to_cpus(node, node_cpus);
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (numa_bitmask_isbitset(node_cpus, c))
            CPU_SET(c, &cpu_set);
    }
    numa_free_cpumask(node_cpus);

    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        // Just warn, don't hard-fail
        std::cerr << "Warning: Failed to set thread affinity for NUMA node " << node << std::endl;
    }
}
#endif

//==============================================================================
//  Low-level helpers
//==============================================================================

namespace detail {

inline void ensure_overwritable(const std::string& path)
{
    if (ACCESS(path.c_str(), F_OK) == 0 &&
        ACCESS(path.c_str(), W_OK) == -1)
        throw std::runtime_error("Refusing to overwrite read-only file: " + path);
}

/*-------------------------------------------------------------------------*/
/*  NUMA / huge-page scratch allocator                                     */
/*-------------------------------------------------------------------------*/

enum class AllocKind : uint8_t { kHuge, kNuma, kPosix };

struct ScratchBuffer {
    uint8_t*   ptr   = nullptr;
    size_t     size  = 0;
    AllocKind  kind  = AllocKind::kPosix;

    ~ScratchBuffer() { release(); }

    void ensure(size_t requiredBytes, int numaNode)
    {
        if (ptr && size >= requiredBytes) return;   // already big enough

        release();                                  // drop previous buffer
        size  = requiredBytes;

#if defined(__linux__)
        /*--- try 2 MiB huge-pages first -----------------------------------*/
        if (requiredBytes >= (2UL << 20)) {
            size_t hpSize = ((requiredBytes + (2UL << 20) - 1) >> 21) << 21;
            void*  hp     = ::mmap(nullptr, hpSize,
                                   PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                                   -1, 0);
            if (hp != MAP_FAILED) {
                /* bind huge-page mapping to NUMA node */
                unsigned long mask = 1UL << numaNode;
                ::mbind(hp, hpSize, MPOL_BIND, &mask,
                        sizeof(mask) * 8, /*flags=*/0);
                ptr  = static_cast<uint8_t*>(hp);
                size = hpSize;
                kind = AllocKind::kHuge;
                return;
            }
        }
        /*--- fall back to libnuma's page-interleaved allocation -----------*/
        ptr = static_cast<uint8_t*>(::numa_alloc_onnode(requiredBytes, numaNode));
        if (ptr) {
            kind = AllocKind::kNuma;
            return;
        }
#endif
        /*--- final fallback: portable aligned heap memory -----------------*/
        constexpr size_t ALIGN = 64;
        if (::posix_memalign(reinterpret_cast<void**>(&ptr), ALIGN, requiredBytes))
            throw std::bad_alloc();
        kind = AllocKind::kPosix;
    }

    void release() noexcept
    {
        if (!ptr) return;
#if defined(__linux__)
        switch (kind)
        {
            case AllocKind::kHuge:  ::munmap(ptr, size);                      break;
            case AllocKind::kNuma:  ::numa_free(ptr, size);                   break;
            case AllocKind::kPosix: std::free(ptr);                           break;
        }
#else
        std::free(ptr);
#endif
        ptr = nullptr;
        size = 0;
    }
};

/* thread-local scratch ----------------------------------------------------*/
static thread_local ScratchBuffer tlsScratch;

/*-------------------------------------------------------------------------*/
/*  Tiny RAII for TIFF*                                                    */
/*-------------------------------------------------------------------------*/
struct TiffHandle {
    TIFF* p = nullptr;
    explicit TiffHandle(const std::string& name, const char* mode) { p = TIFFOpen(name.c_str(), mode); }
    ~TiffHandle() { if (p) TIFFClose(p); }
    explicit operator bool() const noexcept { return p != nullptr; }
    TIFF* get() const noexcept { return p; }
};

} // namespace detail

//==============================================================================
//  Task description
//==============================================================================

struct SliceJob {
    const uint8_t* volumeBase = nullptr;
    size_t         offset     = 0;         // byte offset of this Z-slice
    mwSize         rows       = 0;         // R (first MATLAB dim)
    mwSize         cols       = 0;         // C (second MATLAB dim)
    bool           isXYZ      = true;      // data layout flag
    mxClassID      classId    = mxUINT8_CLASS;
    uint16_t       compTag    = COMPRESSION_NONE;
    size_t         bytesPerSlice = 0;
    size_t         bytesPerPx    = 1;
    size_t         zIndex        = 0;      // for error reporting
    std::string    filename;               // destination TIFF
};

//==============================================================================
//  Per-slice save routine  (runs inside worker threads)
//==============================================================================

static void save_one_slice(const SliceJob& job, int numaNode)
{
    using namespace detail;

    ensure_overwritable(job.filename);

    /*---------------------------------------------------------------*
     * Compute source geometry                                       *
     *---------------------------------------------------------------*/
    const mwSize srcRows = job.isXYZ ? job.cols : job.rows;
    const mwSize srcCols = job.isXYZ ? job.rows : job.cols;
    const uint8_t* src   = job.volumeBase + job.offset;

    /*---------------------------------------------------------------*
     * Open a temporary TIFF (write-then-rename avoids partial files) *
     *---------------------------------------------------------------*/
    const std::string tmpName = job.filename + ".tmp";
    TiffHandle tif(tmpName, "w");
    if (!tif) throw std::runtime_error("Cannot open \"" + tmpName + "\"");

    TIFF* const tf = tif.get();
    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH,       srcCols);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH,      srcRows);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL,  1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE,   (job.bytesPerPx == 2) ? 16 : 8);
    TIFFSetField(tf, TIFFTAG_COMPRESSION,      job.compTag);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC,      PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);
    TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP,     srcRows);

    /*---------------------------------------------------------------*
     * Decide I/O buffer                                             *
     *---------------------------------------------------------------*/
    const uint8_t* ioBuf = nullptr;

    if (job.isXYZ && job.compTag == COMPRESSION_NONE) {
        /*--- fast path: already contiguous in [X Y] order -----------*/
        ioBuf = src;
    } else {
        /*--- slow path: transpose and/or compress ------------------*/
        detail::tlsScratch.ensure(job.bytesPerSlice, numaNode);
        uint8_t* dst = detail::tlsScratch.ptr;

        if (!job.isXYZ) {
            /* transpose [Y X]  →  [X Y] */
            for (mwSize c = 0; c < srcCols; ++c) {
                const uint8_t* colSrc = src + c * job.rows * job.bytesPerPx;
                for (mwSize r = 0; r < srcRows; ++r) {
                    size_t d = (static_cast<size_t>(r) * srcCols + c) * job.bytesPerPx;
                    std::memcpy(dst + d,
                                colSrc + r * job.bytesPerPx,
                                job.bytesPerPx);
                }
            }
        } else {
            std::memcpy(dst, src, job.bytesPerSlice);
        }
        ioBuf = dst;
    }

    /*---------------------------------------------------------------*
     * Write the data                                                *
     *---------------------------------------------------------------*/
    const tsize_t written =
        (job.compTag == COMPRESSION_NONE)
            ? TIFFWriteRawStrip   (tf, 0, const_cast<uint8_t*>(ioBuf), job.bytesPerSlice)
            : TIFFWriteEncodedStrip(tf, 0, const_cast<uint8_t*>(ioBuf), job.bytesPerSlice);

    if (written < 0)
        throw std::runtime_error("TIFF write failed (Z=" +
                                 std::to_string(job.zIndex) + ")");

    /*---------------------------------------------------------------*
     * Commit file                                                   *
     *---------------------------------------------------------------*/
#if defined(__linux__)
    ::unlink(job.filename.c_str());     // ignore ENOENT
#else
    _unlink(job.filename.c_str());
#endif
    if (::rename(tmpName.c_str(), job.filename.c_str()) != 0)
        throw std::runtime_error("rename(" + tmpName + ") failed");
}

//==============================================================================
//  Thread-pool style worker
//==============================================================================

struct JobQueue {
    const std::vector<SliceJob>* jobs = nullptr;
    std::atomic_size_t nextJobIndex {0};
    std::mutex  errorMutex;
    std::vector<std::string> errors;
};

static void worker(JobQueue& queue, int numaNode)
{
    constexpr size_t CHUNK = 16;

#if defined(__linux__)
    pin_thread_to_numa_node(numaNode);  // Critical: ensure NUMA-locality
#endif

    while (true) {
        const size_t start = queue.nextJobIndex.fetch_add(CHUNK, std::memory_order_relaxed);
        if (start >= queue.jobs->size()) break;
        const size_t end = std::min(start + CHUNK, queue.jobs->size());

        for (size_t i = start; i < end; ++i) {
            try {
                save_one_slice((*queue.jobs)[i], numaNode);
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lock(queue.errorMutex);
                queue.errors.emplace_back(ex.what());
            }
        }
    }
}

//==============================================================================
//  Utility: Count physical CPU cores (Linux)
//==============================================================================
static size_t count_physical_cores() {
#if defined(__linux__)
    // Count physical (not logical/HT) cores using /proc/cpuinfo
    std::vector<std::pair<int,int>> unique_pairs;
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return std::thread::hardware_concurrency();
    char line[256];
    int phys_id = -1, core_id = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "physical id\t: %d", &phys_id) == 1) continue;
        if (sscanf(line, "core id\t: %d", &core_id) == 1) {
            unique_pairs.emplace_back(phys_id, core_id);
        }
    }
    fclose(fp);
    // Deduplicate
    std::sort(unique_pairs.begin(), unique_pairs.end());
    unique_pairs.erase(std::unique(unique_pairs.begin(), unique_pairs.end()), unique_pairs.end());
    return unique_pairs.empty() ? std::thread::hardware_concurrency() : unique_pairs.size();
#else
    return std::thread::hardware_concurrency();
#endif
}

//==============================================================================
//  MEX entry point
//==============================================================================

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        // --------------------------------------------------------------
        // Parse input arguments and validate
        // --------------------------------------------------------------
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Usage: save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])");

        const mxArray* volumeMx = prhs[0];
        if (!mxIsUint8(volumeMx) && !mxIsUint16(volumeMx))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Volume must be uint8 or uint16.");

        const mwSize* dims = mxGetDimensions(volumeMx);
        const size_t rows = dims[0];
        const size_t cols = dims[1];
        const size_t numZSlices = (mxGetNumberOfDimensions(volumeMx) == 3) ? dims[2] : 1;

        const bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                           (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        char* compressionCStr  = mxArrayToUTF8String(prhs[3]);
        if (!compressionCStr) mexErrMsgIdAndTxt("save_bl_tif:Input","Bad compression arg");
        const std::string compressionStr(compressionCStr);
        mxFree(compressionCStr);

        const uint16_t compressionTag =
              (compressionStr == "lzw")     ? COMPRESSION_LZW
            : (compressionStr == "deflate") ? COMPRESSION_DEFLATE
            : (compressionStr == "zip")     ? COMPRESSION_DEFLATE
            : (compressionStr == "none")    ? COMPRESSION_NONE
            : throw std::runtime_error("Invalid compression option");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numZSlices)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "fileList must be a 1×Z cell array.");

        std::vector<std::string> filePaths(numZSlices);
        for (size_t k = 0; k < numZSlices; ++k) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], k));
            filePaths[k].assign(s);
            mxFree(s);
        }

        // --------------------------------------------------------------
        // Build job list
        // --------------------------------------------------------------
        const uint8_t* volumeBasePtr = static_cast<const uint8_t*>(mxGetData(volumeMx));
        const mxClassID mxId   = mxGetClassID(volumeMx);
        const size_t    bytesPerPx  = (mxId == mxUINT16_CLASS) ? 2 : 1;
        const size_t    bytesPerSlice = rows * cols * bytesPerPx;

        std::vector<SliceJob> jobs;
        jobs.reserve(numZSlices);
        for (size_t z = 0; z < numZSlices; ++z) {
            jobs.push_back({ volumeBasePtr,
                             z * bytesPerSlice,
                             rows, cols,
                             isXYZ,
                             mxId,
                             compressionTag,
                             bytesPerSlice,
                             bytesPerPx,
                             z,
                             filePaths[z] });
        }

        // --------------------------------------------------------------
        // Determine thread count & NUMA topology
        // --------------------------------------------------------------
        size_t maxThreads = count_physical_cores();
        if (maxThreads == 0) maxThreads = 1;

        if (nrhs == 5) {
            const double requested = mxGetScalar(prhs[4]);
            if (requested > 0)
                maxThreads = std::min(static_cast<size_t>(requested), jobs.size());
        } else {
            maxThreads = std::min(maxThreads, jobs.size());
        }

        // NUMA node count
        size_t numaNodes = 1;
#if defined(__linux__)
        if (numa_available() != -1)
            numaNodes = numa_max_node() + 1;
#endif

        // --------------------------------------------------------------
        // Launch worker threads
        // --------------------------------------------------------------
        JobQueue queue;
        queue.jobs = &jobs;

        std::vector<std::thread> workerThreads;
        workerThreads.reserve(maxThreads);

        for (size_t t = 0; t < maxThreads; ++t) {
            const int node = static_cast<int>(t % numaNodes);
            workerThreads.emplace_back(worker, std::ref(queue), node);
        }
        for (auto& th : workerThreads) th.join();

        // --------------------------------------------------------------
        // Report aggregated errors (if any)
        // --------------------------------------------------------------
        if (!queue.errors.empty()) {
            std::string msg("save_bl_tif encountered errors:\n");
            for (const auto& e : queue.errors) msg += "  • " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        // --------------------------------------------------------------
        // Optional echo-back of the volume
        // --------------------------------------------------------------
        if (nlhs > 0) plhs[0] = const_cast<mxArray*>(volumeMx);
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", ex.what());
    }
}
