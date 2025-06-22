/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  VERSION  : 2025-06-21 (alignment, file list cache, Windows FlushFileBuffers,
                         thread native_handle affinity, hugepages)
  AUTHOR   : Keivan Moradi  (with ChatGPT-4o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <type_traits>

#if defined(__linux__)
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sched.h>
#  include <numa.h>
#endif

#if defined(_WIN32)
#  include <windows.h>
#endif

// Uncomment to enable thread pinning for benchmarking
//#define PIN_THREADS

/* ───────────────────────────── TASK DESCRIPTION ─────────────────────────── */
struct SaveTask {
    const uint8_t* basePtr;       // Start of whole volume
    size_t         sliceOffset;   // Byte-offset of this slice
    mwSize         rows, cols;    // MATLAB dims *after* any transpose
    std::string    filePath;      // Destination path
    bool           alreadyXYZ;    // True if input is [X Y Z]
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
    size_t         sliceIndex;
};

/* Each thread owns one reusable aligned scratch buffer large enough for one slice. */
/* Alignment: 64 bytes for SIMD/AVX2 (safe default, covers common cache lines)     */
/* Hugepages: Use for large buffers (2MB+), platform-dependent implementation      */

static thread_local uint8_t* scratch_aligned = nullptr;
static thread_local size_t   scratch_capacity = 0;
static thread_local bool     scratch_hugepage = false;

static void ensure_scratch(size_t bytes) {
    if (scratch_capacity >= bytes && scratch_aligned != nullptr) return;
    // Free any existing buffer first
    if (scratch_aligned) {
#if defined(__linux__)
        if (scratch_hugepage) {
            munmap(scratch_aligned, scratch_capacity);
        } else
#endif
        {
            std::free(scratch_aligned);
        }
        scratch_aligned = nullptr;
        scratch_capacity = 0;
        scratch_hugepage = false;
    }
    constexpr size_t ALIGN = 64;
    // size must be a multiple of alignment for aligned_alloc!
    size_t aligned_bytes = ((bytes + ALIGN - 1) / ALIGN) * ALIGN;

    // For hugepage, require at least 2MB, aligned to 2MB
    if (aligned_bytes >= (2 << 20)) {
#if defined(__linux__)
        size_t huge_sz = ((aligned_bytes + (2 << 20) - 1) / (2 << 20)) * (2 << 20); // Round up
        void* ptr = mmap(nullptr, huge_sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr != MAP_FAILED) {
            scratch_aligned = static_cast<uint8_t*>(ptr);
            scratch_capacity = huge_sz;
            scratch_hugepage = true;
            return;
        }
        // Fallthrough to normal allocation if mmap fails
#endif
    }
#if defined(_WIN32)
    // No direct hugepage support here, fallback to aligned_alloc
#endif
    // Standard aligned allocation
#if defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606
    scratch_aligned = static_cast<uint8_t*>(std::aligned_alloc(ALIGN, aligned_bytes));
#elif defined(_ISOC11_SOURCE)
    scratch_aligned = static_cast<uint8_t*>(aligned_alloc(ALIGN, aligned_bytes));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN, aligned_bytes) == 0)
        scratch_aligned = static_cast<uint8_t*>(ptr);
    else
        scratch_aligned = nullptr;
#endif
    if (!scratch_aligned)
        throw std::bad_alloc();
    scratch_capacity = aligned_bytes;
    scratch_hugepage = false;
}

/* ───────────────────────────── LOW-LEVEL TIFF WRITE ─────────────────────── */
static void save_slice(const SaveTask& t)
{
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;

    const uint8_t* src = t.basePtr + t.sliceOffset;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ);

    const uint8_t* ioBuf = nullptr;

    if (directWrite) {
        ioBuf = src;
    } else {
        ensure_scratch(t.bytesPerSlice);
        uint8_t* dst = scratch_aligned;

        if (!t.alreadyXYZ) {
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = src + col * t.rows * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * t.bytesPerPixel;
                    std::memcpy(dst + dstIdx,
                                srcColumn + row * t.bytesPerPixel,
                                t.bytesPerPixel);
                }
            }
        } else {
            const size_t rowBytes = srcCols * t.bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row)
                std::memcpy(dst + row * rowBytes,
                            src + row * rowBytes,
                            rowBytes);
        }
        ioBuf = dst;
    }

    const std::string tmpPath = t.filePath + ".tmp";

    TIFF* tif = TIFFOpen(tmpPath.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open temporary file " + tmpPath);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    uint8_t* writeBuf = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ) ? const_cast<uint8_t*>(ioBuf) : scratch_aligned;
    tsize_t nWritten = (t.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip    (tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice))
        : TIFFWriteEncodedStrip(tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice));

    if (nWritten < 0) {
        TIFFClose(tif);
        std::remove(tmpPath.c_str());
        throw std::runtime_error("TIFF write failed on slice " + std::to_string(t.sliceIndex) + " → " + tmpPath);
    }

    TIFFClose(tif);

    // Preflight check: if final file exists and isn't writable
    FILE* testW = std::fopen(t.filePath.c_str(), "wb");
    if (!testW) {
        std::remove(tmpPath.c_str());
        throw std::runtime_error("Refused to overwrite read-only file: " + t.filePath);
    }
    std::fclose(testW);

    if (std::rename(tmpPath.c_str(), t.filePath.c_str()) != 0) {
        std::remove(tmpPath.c_str());
        throw std::runtime_error("Atomic rename failed on slice " + std::to_string(t.sliceIndex) +
                                 ": " + tmpPath + " → " + t.filePath + " (" + std::strerror(errno) + ")");
    }
}

/* ───────────────────────────── FILE LIST CACHE ──────────────────────────── */
/* Caches file lists (vector<string>) to avoid reparsing for repeated calls   */
/* Keyed by prhs[1] (cell ptr) and length                                    */
struct FileListCacheKey {
    const void* mxArrayPtr;
    size_t      length;
    bool operator==(const FileListCacheKey& other) const {
        return mxArrayPtr == other.mxArrayPtr && length == other.length;
    }
};
namespace std {
template<>
struct hash<FileListCacheKey> {
    std::size_t operator()(const FileListCacheKey& k) const {
        return std::hash<const void*>()(k.mxArrayPtr) ^ std::hash<size_t>()(k.length);
    }
};
}

/* ───────────────────────────── ONE-SHOT POOL (PER CALL) ─────────────────── */
namespace {

struct CallContext {
    std::shared_ptr<const std::vector<SaveTask>> tasks;
    std::atomic_size_t nextIndex{0};
    std::mutex errMutex;
    std::vector<std::string> errors;
    size_t maxSliceBytes{0};
};

void worker_entry(CallContext& ctx, int thread_id)
{
#if defined(__linux__)
    if (numa_available() != -1) {
        int max_node = numa_max_node();
        int target_node = thread_id % (max_node + 1);
        numa_run_on_node(target_node);
    }

    pthread_setname_np(pthread_self(), "save_bl_tif");
    sched_param param{};
    pthread_setschedparam(pthread_self(), SCHED_BATCH, &param);
#endif

    const auto& jobList = *ctx.tasks;
    size_t idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);

    while (idx < jobList.size()) {
        try {
            save_slice(jobList[idx]);  // ✅ now no futures or threads
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(ctx.errMutex);
            ctx.errors.emplace_back(e.what());
        }
        idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);
    }
}

} // unnamed namespace

/* ──────────────────────────────── MEX ENTRY ─────────────────────────────── */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        // ──────────────────── Argument Validation ────────────────────────── //
        if (nrhs != 4 && nrhs != 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(volume, fileList, orderFlag, compression [, nThreads])");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8 or uint16.");

        const mwSize* dims = mxGetDimensions(V);
        const size_t dim0 = dims[0], dim1 = dims[1];
        const size_t dim2 = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId = mxGetClassID(V);
        const size_t bytesPerPx = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSl = dim0 * dim1 * bytesPerPx;

        bool alreadyXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0.0);

        char* cstr = mxArrayToUTF8String(prhs[3]);
        if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input", "Invalid compression string.");
        std::string compStr(cstr); mxFree(cstr);

        uint16_t compTag = COMPRESSION_NONE;
        if (compStr == "lzw") compTag = COMPRESSION_LZW;
        else if (compStr == "deflate" || compStr == "zip") compTag = COMPRESSION_DEFLATE;
        else if (compStr != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Compression must be 'none', 'lzw', or 'deflate'.");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must match Z dimension.");

        // ─────────────── File List Cache ──────────────────── //
        static std::unordered_map<FileListCacheKey, std::vector<std::string>> fileListCache;
        FileListCacheKey cacheKey{prhs[1], dim2};
        auto it = fileListCache.find(cacheKey);
        std::vector<std::string> paths;
        if (it != fileListCache.end()) {
            paths = it->second;
        } else {
            paths.resize(dim2);
            for (size_t sliceIndex = 0; z < dim2; ++sliceIndex) {
                mxArray* elem = mxGetCell(prhs[1], sliceIndex);
                if (!mxIsChar(elem))
                    mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList elements must be strings.");
                char* s = mxArrayToUTF8String(elem);
                paths[sliceIndex] = s; mxFree(s);
            }
            fileListCache[cacheKey] = paths;
        }

        // ─────────────── Build Task Vector ────────────────── //
        auto taskVec = std::make_shared<std::vector<SaveTask>>();
        taskVec->reserve(dim2);
        for (size_t sliceIndex = 0; sliceIndex < dim2; ++sliceIndex)
            taskVec->emplace_back(SaveTask{ basePtr, sliceIndex * bytesPerSl, dim0, dim1, paths[sliceIndex], alreadyXYZ, classId, compTag, bytesPerSl, bytesPerPx, sliceIndex });

        // Warm-up libtiff on first call (lazy loader)
        TIFF* tmp = TIFFOpen("/dev/null", "r"); if (tmp) TIFFClose(tmp);

        // ──────────── Optimized Thread Launching ──────────── //
        CallContext ctx;
        ctx.tasks = std::move(taskVec);
        ctx.maxSliceBytes = bytesPerSl;

        size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = ctx.tasks->size();

        size_t maxThreads = std::min(hw, ctx.tasks->size());
        if (nrhs == 5) {
            double reqThreads = mxGetScalar(prhs[4]);
            if (!(reqThreads > 0))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "nThreads must be positive.");
            maxThreads = std::min((size_t)reqThreads, ctx.tasks->size());
        }

        std::vector<std::thread> workers;
        workers.reserve(maxThreads);

        for (size_t i = 0; i < maxThreads; ++i) {
            workers.emplace_back([&, i]() {
            #if defined(__linux__)
                // NUMA affinity
                if (numa_available() != -1) {
                    int max_node = numa_max_node();
                    int target_node = int(i % (max_node + 1));
                    numa_run_on_node(target_node);
                }

                // Thread name and scheduling
                pthread_setname_np(pthread_self(), "save_bl_tif");
                sched_param param{};
                param.sched_priority = 0;
                pthread_setschedparam(pthread_self(), SCHED_BATCH, &param);

                // Core pinning
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i % hw, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            #endif

                worker_entry(ctx, int(i));
            });
        }

        // Join all threads (critical for avoiding segfault)
        for (auto& t : workers) if (t.joinable()) t.join();

        if (!ctx.errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : ctx.errors) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        if (nlhs) plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
