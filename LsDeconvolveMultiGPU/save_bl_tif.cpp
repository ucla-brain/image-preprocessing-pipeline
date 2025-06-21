/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  Author        : Keivan Moradi  (with ChatGPT-4o assistance)
  License       : GNU GPL v3   <https://www.gnu.org/licenses/>

  OVERVIEW
  -------
  • Purpose
      Save every Z-slice of a 3-D volume to its own TIFF file, optionally
      compressed (LZW or Deflate). Supports MATLAB’s default [Y X Z] layout
      and the alternative [X Y Z] layout.

  • Key features
      – Accepts uint8 or uint16 input.
      – Cross-platform: Windows, Linux, macOS (libtiff backend).
      – Reusable std::thread pool (one thread per HW core, min 8).
      – 1 MiB thread stacks on glibc ≥ 2.34 to minimise VM usage.
      – Thread-local scratch buffer allocated **once** per worker.
      – Uses TIFFWriteRawStrip when compression="none" (no extra memcpy/CRC)
      – AVX-2 16×16 blocked transpose for [Y X Z] slices when CPU supports it
      – Fast path for [Y X Z] avoids per-pixel multiplications.
      – Matches `load_bl_tif.cpp` for slice order and dimensions.

  PARALLELISM
  -----------
  • Atomic index dispatch:
        next = g_next.fetch_add(1);          // lock-free task claim
    Workers loop until all slices are processed.
  • Task vector is held in a `shared_ptr`, eliminating race conditions at
    batch teardown.
  • Pool is created once per MATLAB session and destroyed via `mexAtExit`.

  MEMORY
  ------
  • Each worker owns a thread-local `std::vector<uint8_t>` large enough for the
    biggest slice seen so far; no further reallocations on later calls.
  • Total additional RAM ≈ (#threads × slice_bytes) + O(Z).

  USAGE
  -----
      save_bl_tif(volume3d, fileList, orderFlag, compression)

      volume3d    : 3-D uint8 | uint16 array
      fileList    : 1×Z cell array of output paths (char or string)
      orderFlag   : logical or uint32
                      true  → input is [X Y Z]
                      false → input is [Y X Z]  (MATLAB default)
      compression : "none" | "lzw" | "deflate"

  ==============================================================================*/

#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include "transpose_avx2.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
  #include <pthread.h>
  #include <fcntl.h>     // posix_fadvise
  #include <unistd.h>
#endif

/* --- runtime AVX-2 check (GCC/Clang) ----------------------------------- */
#if defined(__x86_64__) && defined(__AVX2__)
#   if defined(__GNUC__) || defined(__clang__)
    static bool have_avx2 = __builtin_cpu_supports("avx2");
#   else   /* MSVC */
    #include <intrin.h>
    static bool have_avx2 = ([]{
        int cpuInfo[4];
        __cpuid(cpuInfo, 7);
        return (cpuInfo[1] & (1 << 5)) != 0;   // EBX bit 5 = AVX2
    })();
#   endif
#else
    static const bool have_avx2 = false;
#endif


/*==============================================================================
   RAII wrapper for MATLAB strings
==============================================================================*/
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* a) : ptr(mxArrayToUTF8String(a)) {
        if (!ptr)
            mexErrMsgIdAndTxt("save_bl_tif:BadString",
                              "mxArrayToUTF8String returned null");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    MatlabString(const MatlabString&)            = delete;
    MatlabString& operator=(const MatlabString&) = delete;
};

/*==============================================================================
    Task description – one per Z slice
==============================================================================*/
struct SaveTask {
    const uint8_t* base;
    mwSize         dim0;   // Y
    mwSize         dim1;   // X
    mwSize         z;      // slice index
    std::string    path;
    bool           isXYZ;
    mxClassID      classId;  // uint8/uint16
    std::string    comp;     // "none" | "lzw" | "deflate"
};

/*==============================================================================
    Per-slice save – now with raw-strip + fadvise
==============================================================================*/
static thread_local std::vector<uint8_t> scratch;   // reused by thread

static void save_slice(const SaveTask& t)
{
    const size_t es = (t.classId == mxUINT16_CLASS ? 2 : 1);
    const mwSize width  = t.isXYZ ? t.dim0 : t.dim1;
    const mwSize height = t.isXYZ ? t.dim1 : t.dim0;
    const size_t sliceOff = static_cast<size_t>(t.z) * t.dim0 * t.dim1;
    const size_t bytes    = static_cast<size_t>(width) * height * es;

    /* Allocate scratch buffer once per worker (grow-only) */
    if (scratch.capacity() < bytes) {
        scratch.reserve(bytes);
        scratch.resize(bytes);
    }
    uint8_t* buf = scratch.data();

    /* -------- transpose into buf -------- */
    if (!t.isXYZ)   /* fast Y-X-Z path */
    {
        if (have_avx2 && (width & 15) == 0 && (height & 15) == 0)
        {
            /* tile-wise 16×16 SIMD transpose */
            for (mwSize y0 = 0; y0 < height; y0 += 16) {
                for (mwSize x0 = 0; x0 < width; x0 += 16) {
                    const uint8_t* sp =
                        t.base + (sliceOff + static_cast<size_t>(x0) * t.dim0
                                  + y0) * es;
                    uint8_t* dp = buf + (static_cast<size_t>(y0) * width + x0) * es;
                    if (es == 1)
                        simd::transpose16x16_u8(sp, es * t.dim0, dp, es * width);
                    else
                        simd::transpose16x16_u16((const uint16_t*)sp, t.dim0,
                                                 (uint16_t*)dp, width);
                }
            }
        }
        else
        {
            /* scalar column-wise copy (unchanged) */
            for (mwSize x = 0; x < width; ++x) {
                const uint8_t* srcCol =
                    t.base + (sliceOff + static_cast<size_t>(x) * t.dim0) * es;
                for (mwSize y = 0; y < height; ++y) {
                    size_t dstIdx = (static_cast<size_t>(y) * width + x) * es;
                    const uint8_t* src = srcCol + static_cast<size_t>(y) * es;
                    if (es == 1) buf[dstIdx] = *src;
                    else         std::memcpy(buf + dstIdx, src, 2);
                }
            }
        }
    }
    else   /* X-Y-Z: each row already contiguous */
    {
        const size_t rowBytes = width * es;
        for (mwSize y = 0; y < height; ++y) {
            const uint8_t* srcRow =
                t.base + (sliceOff + static_cast<size_t>(y) * t.dim0) * es;
            std::memcpy(buf + static_cast<size_t>(y) * rowBytes,
                        srcRow, rowBytes);
        }
    }

    /* -------- write entire slice in one strip -------- */
    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (es == 2 ? 16 : 8));

    const bool noComp = (t.comp == "none");
    uint16_t compTag  = noComp ? COMPRESSION_NONE :
                       (t.comp == "lzw") ? COMPRESSION_LZW : COMPRESSION_DEFLATE;
    TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);  // one strip

    const tsize_t wrote = noComp
        ? TIFFWriteRawStrip    (tif, 0, buf, bytes)
        : TIFFWriteEncodedStrip(tif, 0, buf, bytes);

    if (wrote < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF write failed on " + t.path);
    }

#if defined(__linux__)
    /* Hint kernel: drop file pages from cache immediately */
    int fd = TIFFFileno(tif);
    if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    TIFFClose(tif);
}

/*==============================================================================
    Thread-pool internals (unchanged – but with shared_ptr fix)
==============================================================================*/
namespace {

static std::shared_ptr<const std::vector<SaveTask>> g_tasks;
static std::atomic_size_t           g_next{0};
static std::atomic_size_t           g_remaining{0};
static std::vector<std::string>     g_errs;

static std::mutex                   g_mtx;
static std::condition_variable      g_cv;
static bool                         g_stop = false;
static std::vector<std::thread>     g_workers;

/* shrink default pthread stack to 1 MiB (Linux/glibc ≥2.34) */
#if defined(__linux__)
inline void shrink_default_stack()
{
    static bool done = false;
    if (done) return;
    done = true;
  #if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2,34)
    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
  #endif
}
#endif

static void worker_loop()
{
    for (;;) {
        std::unique_lock<std::mutex> ul(g_mtx);
        g_cv.wait(ul, [] { return g_tasks || g_stop; });
        if (g_stop) return;

        auto tasks = g_tasks;  // retain vector
        ul.unlock();

        size_t idx = g_next.fetch_add(1);
        while (idx < tasks->size()) {
            try { save_slice((*tasks)[idx]); }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(g_mtx);
                g_errs.emplace_back(e.what());
            }
            if (g_remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(g_mtx);
                g_tasks.reset();
                g_cv.notify_all();
            }
            idx = g_next.fetch_add(1);
        }
    }
}

inline void ensure_pool(size_t requested)
{
    if (!g_workers.empty()) return;

#if defined(__linux__)
    shrink_default_stack();
#endif
    requested = std::max<size_t>(8, std::thread::hardware_concurrency()); // user-approved: all hw threads, ≥8

    g_workers.reserve(requested);
    try {
        for (size_t i = 0; i < requested; ++i)
            g_workers.emplace_back(worker_loop);
    } catch (...) {
        g_stop = true;
        g_cv.notify_all();
        for (auto& t : g_workers)
            if (t.joinable()) t.join();
        throw;
    }

    mexAtExit(+[] {
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_stop = true;
        }
        g_cv.notify_all();
        for (auto& t : g_workers)
            if (t.joinable()) t.join();
    });
}

} // anonymous namespace

/*==============================================================================
    MEX gateway
==============================================================================*/
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        /* -------- argument validation (unchanged) -------- */
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");
        if (mxGetNumberOfDimensions(V) != 3)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be 3-D");

        const mwSize dim0 = mxGetDimensions(V)[0];
        const mwSize dim1 = mxGetDimensions(V)[1];
        const mwSize dim2 = mxGetDimensions(V)[2];

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        const bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                          ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
                           *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString cs(prhs[3]);
        std::string  comp(cs.get());
        if (comp != "none" && comp != "lzw" && comp != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        /* -------- build task vector -------- */
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        auto taskVec = std::make_shared<std::vector<SaveTask>>();
        taskVec->reserve(dim2);

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* c = mxGetCell(prhs[1], z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString path(c);
            taskVec->push_back({base, dim0, dim1, z, path.get(),
                                isXYZ, mxGetClassID(V), comp});
        }

        if (taskVec->empty()) return;

        /* -------- hand tasks to pool -------- */
        ensure_pool(taskVec->size());    // start workers once

        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_tasks     = taskVec;
            g_next      = 0;
            g_remaining = taskVec->size();
            g_errs.clear();
        }
        g_cv.notify_all();

        /* wait for batch completion */
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv.wait(lk, [] { return !g_tasks; });
        }

        if (!g_errs.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (auto& e : g_errs) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
