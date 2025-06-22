/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  USAGE:
    save_bl_tif(volume, fileList, orderXYZ, compression, [nThreads])

    - volume     : 3-D uint8 or uint16 array, [X Y Z] or [Y X Z] (see orderXYZ)
    - fileList   : cell array of strings, one filename per Z-slice
    - orderXYZ   : logical or numeric; true if already [X Y Z], else [Y X Z]
    - compression: 'none', 'lzw', or 'deflate'
    - nThreads   : (optional) max number of threads to use (default: hardware_concurrency)

  FEATURES:
    - Each Z-slice is saved to its own TIFF file (optionally compressed).
    - Ultra-high parallel throughput; optimized for multi-socket NUMA servers.
    - Safe overwrite checks (won't clobber read-only files).
    - Robust error reporting.
    - No memory leaks: all per-thread resources are released.
    - Automatically uses hugepages and NUMA locality if available (Linux).

  LIMITATIONS:
    - Large numbers of threads may allocate significant RAM.
    - Make sure per-slice buffer size fits into RAM across all threads.
    - Does not support RGB, only grayscale uint8/uint16.

  AUTHOR   : Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <type_traits>

#if defined(__linux__)
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/uio.h>
#  include <sched.h>
#  include <numa.h>
#  include <numaif.h>
#  define ACCESS access
#endif

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#  define ACCESS _access
#endif

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline void guard_overwrite_writable(const std::string& path) {
    if (ACCESS(path.c_str(), F_OK) == 0) {
        if (ACCESS(path.c_str(), W_OK) == -1) {
#if defined(_WIN32)
            if (errno == EACCES || errno == EPERM)
#else
            if (errno == EACCES)
#endif
                throw std::runtime_error("Refused to overwrite read-only file: " + path);
        }
    }
}

void* alloc_on_node(size_t bytes, int node, bool wantHuge) {
    void* p = nullptr;
#if defined(__linux__)
    if (wantHuge && bytes >= (2UL << 20)) {
        size_t hugeSz = ((bytes + (2UL << 20) - 1) >> 21) << 21;
        void* hp = ::mmap(nullptr, hugeSz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (hp != MAP_FAILED) {
            unsigned long nodemask = 1UL << node;
            ::mbind(hp, hugeSz, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
            return hp;
        }
    }
    p = ::numa_alloc_onnode(bytes, node);
    if (p) return p;
#endif
    constexpr size_t ALIGN = 64;
    if (::posix_memalign(&p, ALIGN, ((bytes + ALIGN - 1) / ALIGN) * ALIGN) != 0)
        p = nullptr;
    return p;
}

void free_on_node(void* ptr, size_t bytes, bool huge) {
#if defined(__linux__)
    if (!ptr) return;
    if (huge) { ::munmap(ptr, bytes); return; }
    ::numa_free(ptr, bytes);
#else
    (void)bytes; (void)huge; std::free(ptr);
#endif
}

// ============================================================================
// STRUCT DEFINITIONS
// ============================================================================

struct SaveTask {
    const uint8_t* basePtr;
    size_t sliceOffset;
    mwSize rows, cols;
    std::string filePath;
    bool alreadyXYZ;
    mxClassID classId;
    uint16_t compressionTag;
    size_t bytesPerSlice;
    size_t bytesPerPixel;
    size_t sliceIndex;
};

struct ScratchBufferRAII {
    uint8_t* buf = nullptr;
    size_t bytes = 0;
    bool huge = false;
    ~ScratchBufferRAII() { if (buf) free_on_node(buf, bytes, huge); }
    void ensure(size_t wantBytes, int numaNode) {
        if (buf && bytes >= wantBytes) return;
        if (buf) free_on_node(buf, bytes, huge);
        buf = static_cast<uint8_t*>(alloc_on_node(wantBytes, numaNode, true));
        huge = (buf && wantBytes >= (2UL << 20) && ((uintptr_t)buf & ((2UL << 20) - 1)) == 0);
        if (!buf) throw std::bad_alloc();
        bytes = wantBytes;
    }
};

static thread_local ScratchBufferRAII tlsScratch;

void save_slice(const SaveTask& t, int numaNode) {
    guard_overwrite_writable(t.filePath);
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;
    const uint8_t* src = t.basePtr + t.sliceOffset;

    std::string tmpPath = t.filePath + ".tmp";
    TIFF* tif = TIFFOpen(tmpPath.c_str(), "w");
    if (!tif) throw std::runtime_error("Failed to open temporary TIFF file: " + tmpPath);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (t.bytesPerPixel == 2) ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    const uint8_t* ioBuf = nullptr;
    if (t.alreadyXYZ && t.compressionTag == COMPRESSION_NONE) {
        ioBuf = src;
    } else {
        tlsScratch.ensure(t.bytesPerSlice, numaNode);
        uint8_t* dst = tlsScratch.buf;
        if (!t.alreadyXYZ) {
            for (mwSize c = 0; c < srcCols; ++c) {
                const uint8_t* colSrc = src + c * t.rows * t.bytesPerPixel;
                for (mwSize r = 0; r < srcRows; ++r) {
                    size_t d = (size_t(r) * srcCols + c) * t.bytesPerPixel;
                    std::memcpy(dst + d, colSrc + r * t.bytesPerPixel, t.bytesPerPixel);
                }
            }
        } else {
            std::memcpy(dst, src, t.bytesPerSlice);
        }
        ioBuf = dst;
    }

    tsize_t written = (t.compressionTag == COMPRESSION_NONE)
                      ? TIFFWriteRawStrip(tif, 0, const_cast<uint8_t*>(ioBuf), t.bytesPerSlice)
                      : TIFFWriteEncodedStrip(tif, 0, const_cast<uint8_t*>(ioBuf), t.bytesPerSlice);
    if (written < 0) throw std::runtime_error("TIFF write failed at slice " + std::to_string(t.sliceIndex));

    TIFFClose(tif);
    ::unlink(t.filePath.c_str());
    if (::rename(tmpPath.c_str(), t.filePath.c_str()) != 0)
        throw std::runtime_error("rename failed: " + tmpPath + " -> " + t.filePath);
}

struct ThreadedSaveContext {
    const std::vector<SaveTask>* allTasks;
    std::atomic_size_t globalTaskIndex{0};
    std::mutex errorMutex;
    std::vector<std::string> errors;
};

void save_worker(ThreadedSaveContext& ctx, int threadId, int numaNode) {
    constexpr size_t CHUNK = 16;
    while (true) {
        size_t start = ctx.globalTaskIndex.fetch_add(CHUNK, std::memory_order_relaxed);
        if (start >= ctx.allTasks->size()) break;
        size_t end = std::min(start + CHUNK, ctx.allTasks->size());
        for (size_t i = start; i < end; ++i) {
            try {
                save_slice((*ctx.allTasks)[i], numaNode);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(ctx.errorMutex);
                ctx.errors.emplace_back(e.what());
            }
        }
    }
}

// ============================================================================
// MEX ENTRY POINT
// ============================================================================

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs != 4 && nrhs != 5)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Usage: save_bl_tif(volume, fileList, orderXYZ, compression [, nThreads])");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16.");

        const mwSize* dims = mxGetDimensions(V);
        size_t R = dims[0], C = dims[1];
        size_t Z = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        bool alreadyXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                          (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        char* cstr = mxArrayToUTF8String(prhs[3]);
        if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input", "Bad compression arg");
        std::string compStr(cstr); mxFree(cstr);
        uint16_t compTag = (compStr == "lzw")     ? COMPRESSION_LZW :
                           (compStr == "deflate") ? COMPRESSION_DEFLATE :
                           (compStr == "zip")      ? COMPRESSION_DEFLATE :
                           (compStr == "none")     ? COMPRESSION_NONE :
                                                     throw std::runtime_error("Invalid compression");
        if (compStr != "none" && compTag == COMPRESSION_NONE)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Compression must be none/lzw/deflate");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != Z)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must match Z dim");

        std::vector<std::string> paths(Z);
        for (size_t k = 0; k < Z; ++k) {
            char* p = mxArrayToUTF8String(mxGetCell(prhs[1], k));
            paths[k] = p; mxFree(p);
        }

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        mxClassID id = mxGetClassID(V);
        size_t bpp = (id == mxUINT16_CLASS) ? 2 : 1;
        size_t bpslice = R * C * bpp;

        std::vector<SaveTask> tasks;
        for (size_t z = 0; z < Z; ++z)
            tasks.push_back(SaveTask{ base, z * bpslice, R, C, paths[z],
                                      alreadyXYZ, id, compTag, bpslice, bpp, z });

        ThreadedSaveContext ctx;
        ctx.allTasks = &tasks;

        size_t hwThreads = std::thread::hardware_concurrency();
        if (hwThreads == 0) hwThreads = 1;
        size_t nThreads = std::min(hwThreads, tasks.size());
        if (nrhs == 5) {
            double d = mxGetScalar(prhs[4]);
            if (d > 0) nThreads = std::min((size_t)d, tasks.size());
        }

        size_t numaNodes = 0;
#if defined(__linux__)
        if (numa_available() != -1)
            numaNodes = numa_max_node() + 1;
#endif

        std::vector<std::thread> workers;
        for (size_t i = 0; i < nThreads; ++i) {
            int node = (numaNodes > 0) ? int(i % numaNodes) : 0;
            workers.emplace_back(save_worker, std::ref(ctx), int(i), node);
        }
        for (auto& t : workers) t.join();

        if (!ctx.errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (auto& e : ctx.errors) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        if (nlhs) plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
