/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (1 TIFF per Z-slice)

  Author:       Keivan Moradi (in collaboration with ChatGPT-4o)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)

  OVERVIEW
  -------
  • Purpose:
      Efficiently saves each Z-slice from a 3D MATLAB array to a separate TIFF
      file using LZW, Deflate, or no compression. Supports [X Y Z] or [Y X Z] input.

  • Highlights:
      – Accepts `uint8` or `uint16` MATLAB input arrays.
      – Fully cross-platform (uses libtiff).
      – Supports multithreading.
      – Compression: none, lzw, or deflate.
      – Matches `load_bl_tif.cpp` slice order and dimensions.

  PARALLELISM
  -----------
  • Parallel I/O is implemented using atomic index dispatching:
      Each worker thread atomically claims the next available task index
      using `std::atomic<size_t>::fetch_add`, avoiding locks or queues.
      This model scales efficiently for uniform workloads like per-slice saves.

  USAGE
  -----
      save_bl_tif(array3d, fileList, orderFlag, compression)

      • array3d      : 3D numeric array, uint8 or uint16
      • fileList     : 1xZ cell array of full path strings
      • orderFlag    : (logical or uint32 scalar)
                         true  = [X Y Z] input
                         false = [Y X Z] input (MATLAB default)
      • compression  : string: "none", "lzw", or "deflate"

  ==============================================================================*/

#include "mex.h"
#include "matrix.h"
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

#if defined(__linux__)
  #include <pthread.h>
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
    mwSize         dim0;      // Y
    mwSize         dim1;      // X
    mwSize         z;         // slice index
    std::string    path;
    bool           isXYZ;     // true => [X Y Z] memory layout
    mxClassID      classId;   // uint8 / uint16
    std::string    comp;      // "none" | "lzw" | "deflate"
};

/*==============================================================================
    Per-slice save with all approved optimisations
==============================================================================*/
static thread_local std::vector<uint8_t> scratch;   // reused by thread

static void save_slice(const SaveTask& t)
{
    const size_t es = (t.classId == mxUINT16_CLASS ? 2 : 1);
    const mwSize width  = t.isXYZ ? t.dim0 : t.dim1;   // X
    const mwSize height = t.isXYZ ? t.dim1 : t.dim0;   // Y
    const size_t sliceOff = static_cast<size_t>(t.z) * t.dim0 * t.dim1;
    const size_t bytes = static_cast<size_t>(width) * height * es;

    /* ensure thread-local buffer large enough */
    scratch.resize(bytes);
    uint8_t* buf = scratch.data();

    /* ---------------- transpose into buf ---------------- */
    if (!t.isXYZ)   /* fast Y-X-Z path (MATLAB default) */
    {
        /* Each MATLAB column (X) is already contiguous in memory.
           Copy one column at a time into row-major destination. */
        for (mwSize x = 0; x < width; ++x) {
            const uint8_t* srcCol =
                t.base + (sliceOff + static_cast<size_t>(x) * t.dim0) * es;
            for (mwSize y = 0; y < height; ++y) {
                size_t dstIdx = (static_cast<size_t>(y) * width + x) * es;
                const uint8_t* src = srcCol + static_cast<size_t>(y) * es;
                if (es == 1)
                    buf[dstIdx] = *src;
                else
                    std::memcpy(buf + dstIdx, src, 2);   // uint16
            }
        }
    }
    else            /* X-Y-Z layout – keep original loops but hoist multiply */
    {
        for (mwSize y = 0; y < height; ++y) {
            const size_t rowDstBase = static_cast<size_t>(y) * width * es;
            const size_t colBase = static_cast<size_t>(y) * t.dim0;  // for srcCol*dim0
            for (mwSize x = 0; x < width; ++x) {
                size_t srcIdx = x + colBase + sliceOff; // srcRow (x) + srcCol(y)*dim0
                size_t dstIdx = rowDstBase + static_cast<size_t>(x) * es;
                if (es == 1)
                    buf[dstIdx] = t.base[srcIdx];
                else
                    std::memcpy(buf + dstIdx, t.base + srcIdx * es, 2);
            }
        }
    }

    /* ---------------- write whole slice in one strip ---------------- */
    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif)
        throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (es == 2 ? 16 : 8));

    uint16_t compTag = (t.comp == "lzw")     ? COMPRESSION_LZW
                     : (t.comp == "deflate") ? COMPRESSION_DEFLATE
                     :                         COMPRESSION_NONE;
    TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);    // 1 strip

    if (TIFFWriteEncodedStrip(tif, 0, buf, bytes) < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFFWriteEncodedStrip failed on " + t.path);
    }
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
