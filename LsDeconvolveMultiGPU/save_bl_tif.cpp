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
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <cstdlib>
#include <cstdio>

#if defined(__linux__)
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <pthread.h>
#  include <sched.h>
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
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;    // If transpose needed
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;

    const uint8_t* src = t.basePtr + t.sliceOffset;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ);

    const uint8_t* ioBuf = nullptr;          // Buffer actually passed to libtiff

    /* 1. Prepare buffer (handle transpose / compression fast-path) */
    if (directWrite) {
        ioBuf = src;                         // Zero-copy path 🚀
    } else {
        ensure_scratch(t.bytesPerSlice);
        uint8_t* dst = scratch_aligned;

        if (!t.alreadyXYZ) {                 // Transpose [Y X] → [X Y]
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = src + col * t.rows * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * t.bytesPerPixel;
                    std::memcpy(dst + dstIdx,
                                srcColumn + row * t.bytesPerPixel,
                                t.bytesPerPixel);
                }
            }
        } else {                            // Already XYZ – need copy only for compression
            const size_t rowBytes = srcCols * t.bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row)
                std::memcpy(dst + row * rowBytes,
                            src + row * rowBytes,
                            rowBytes);
        }
        ioBuf = dst;
    }

    /* 2. Open TIFF (write-mode) */
    TIFF* tif = TIFFOpen(t.filePath.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.filePath);

    /* 3. Required baseline tags */
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,     t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

    /* 4. One-strip-per-image (fast and simple) */
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    /* 5. Write the single strip */
    uint8_t* writeBuf = const_cast<uint8_t*>(ioBuf);
    tsize_t  nWritten = (t.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip    (tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice))
        : TIFFWriteEncodedStrip(tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice));

    if (nWritten < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF write failed on " + t.filePath);
    }

#if defined(__linux__)
    int fd = TIFFFileno(tif);
    if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // Drop page cache
#endif

#if defined(_WIN32)
    // Flush file buffers for async I/O flush on Windows (for very large batch jobs or non-SSD)
    HANDLE hFile = (HANDLE)_get_osfhandle(TIFFFileno(tif));
    if (hFile != INVALID_HANDLE_VALUE) FlushFileBuffers(hFile);
#endif

    TIFFClose(tif);
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

void worker_entry(CallContext& ctx)
{
    // Thread affinity is handled at thread creation (not here)
    const auto& jobList = *ctx.tasks;
    size_t idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);

    while (idx < jobList.size()) {
        try {
            save_slice(jobList[idx]);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(ctx.errMutex);
            ctx.errors.emplace_back(e.what());
        }
        idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);
    }
}

} // unnamed namespace

/* ──────────────────────────────── MEX ENTRY ─────────────────────────────── */
/* ... [rest of includes and code above unchanged] ... */

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        /* ─── 1. ARGUMENT VALIDATION ─────────────────────────────────────── */
        if (nrhs != 4 && nrhs != 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(volume, fileList, orderFlag, compression [, nThreads])");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Volume must be uint8 or uint16.");

        const mwSize* dims   = mxGetDimensions(V);
        const size_t  dim0   = dims[0];
        const size_t  dim1   = dims[1];
        const size_t  dim2   = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* basePtr  = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId = mxGetClassID(V);
        const size_t bytesPerPx = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSl = dim0 * dim1 * bytesPerPx;

        bool alreadyXYZ = false;
        const mxArray* ord = prhs[2];
        if (mxIsLogicalScalar(ord)) alreadyXYZ = mxIsLogicalScalarTrue(ord);
        else if (mxIsNumeric(ord))  alreadyXYZ = (mxGetScalar(ord) != 0.0);

        std::string compStr;
        {
            char* cstr = mxArrayToUTF8String(prhs[3]);
            if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input", "Invalid compression string.");
            compStr = cstr; mxFree(cstr);
        }
        uint16_t compTag = COMPRESSION_NONE;
        if (compStr == "lzw")           compTag = COMPRESSION_LZW;
        else if (compStr == "deflate" || compStr == "zip")
                                         compTag = COMPRESSION_DEFLATE;
        else if (compStr != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Compression must be 'none', 'lzw', or 'deflate'.");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "fileList must be a cell array matching Z dimension.");

        // ── FILE LIST CACHE ──
        static std::unordered_map<FileListCacheKey, std::vector<std::string>> fileListCache;
        FileListCacheKey cacheKey{prhs[1], dim2};
        std::vector<std::string>* cachedPaths = nullptr;
        auto it = fileListCache.find(cacheKey);
        if (it != fileListCache.end()) {
            cachedPaths = &it->second;
        }
        std::vector<std::string> paths;
        if (cachedPaths) {
            paths = *cachedPaths;
        } else {
            mxArray** cellPtr = static_cast<mxArray**>(mxGetData(prhs[1]));
            paths.resize(dim2);
            for (size_t z = 0; z < dim2; ++z) {
                if (!mxIsChar(cellPtr[z]))
                    mexErrMsgIdAndTxt("save_bl_tif:Input",
                        "fileList element is not a string.");
                char* s = mxArrayToUTF8String(cellPtr[z]);
                paths[z] = s; mxFree(s);
            }
            fileListCache[cacheKey] = paths;
        }

        if (paths.empty()) return;

        /* ─── 2. BUILD TASK VECTOR ───────────────────────────────────────── */
        auto taskVec = std::make_shared<std::vector<SaveTask>>();
        taskVec->reserve(dim2);
        for (size_t z = 0; z < dim2; ++z)
            taskVec->push_back({ basePtr,
                                 z * bytesPerSl,
                                 dim0, dim1,
                                 paths[z],
                                 alreadyXYZ,
                                 classId,
                                 compTag,
                                 bytesPerSl,
                                 bytesPerPx });

        /* ─── 3. EAGER-BIND libtiff symbols (one quick call) ─────────────── */
        {
            TIFF* tmp = TIFFOpen("/dev/null", "r");
            if (tmp) TIFFClose(tmp);
        }

        /* ─── 4. LAUNCH ONE-SHOT THREAD POOL ─────────────────────────────── */
        CallContext ctx;
        ctx.tasks          = std::move(taskVec);
        ctx.maxSliceBytes  = bytesPerSl;

        // Optional thread count argument
        size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = ctx.tasks->size();

        size_t nThreads = std::min(hw, ctx.tasks->size());
        if (nrhs == 5) {
            double reqThreads = mxGetScalar(prhs[4]);
            if (!(reqThreads > 0 && reqThreads <= ctx.tasks->size()))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "Invalid nThreads value.");
            nThreads = std::min((size_t)reqThreads, ctx.tasks->size());
        }

        std::vector<std::thread> workers;
        workers.reserve(nThreads);

#ifdef PIN_THREADS
        std::vector<int> coreIds;
        for (int i = 0; i < int(hw); ++i) coreIds.push_back(i);
#endif

        for (size_t i = 0; i < nThreads; ++i) {
            workers.emplace_back(worker_entry, std::ref(ctx));
#ifdef PIN_THREADS
#if defined(__linux__)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(coreIds[i % coreIds.size()], &cpuset);
            pthread_setaffinity_np(workers.back().native_handle(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
            DWORD_PTR mask = DWORD_PTR(1) << (coreIds[i % coreIds.size()]);
            SetThreadAffinityMask(workers.back().native_handle(), mask);
#endif
#endif
        }

        for (auto& t : workers) t.join();

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
