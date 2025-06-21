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
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
  #include <pthread.h>
#endif

/*==============================================================================
    Helper RAII for MATLAB strings
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
    mwSize         dim0;
    mwSize         dim1;
    mwSize         z;
    std::string    path;
    bool           isXYZ;
    mxClassID      classId;
    std::string    comp;
};

/*==============================================================================
    Per-slice save implementation (unchanged)
==============================================================================*/
static void save_slice(const SaveTask& t)
{
    const size_t es = (t.classId == mxUINT16_CLASS ? 2 : 1);

    const mwSize width  = t.isXYZ ? t.dim0 : t.dim1;
    const mwSize height = t.isXYZ ? t.dim1 : t.dim0;
    const size_t sliceOff = static_cast<size_t>(t.z) * t.dim0 * t.dim1;

    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

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
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);

    std::vector<uint8_t> scan(width * es);

    for (mwSize y = 0; y < height; ++y) {
        for (mwSize x = 0; x < width; ++x) {
            mwSize srcRow = t.isXYZ ? x : y;
            mwSize srcCol = t.isXYZ ? y : x;
            size_t srcIdx = srcRow + srcCol * t.dim0 + sliceOff;
            std::memcpy(&scan[x * es], t.base + srcIdx * es, es);
        }
        if (TIFFWriteScanline(tif, scan.data(), y, 0) < 0) {
            TIFFClose(tif);
            throw std::runtime_error("TIFFWriteScanline failed on " + t.path);
        }
    }
    TIFFClose(tif);
}

/*==============================================================================
    Thread-pool internals (persistent across MEX calls)
==============================================================================*/
namespace {

static std::shared_ptr<const std::vector<SaveTask>> g_tasks; // nullptr when idle
static std::atomic_size_t           g_next{0};
static std::atomic_size_t           g_remaining{0};
static std::vector<std::string>     g_errs;

static std::mutex                   g_mtx;
static std::condition_variable      g_cv;
static bool                         g_stop = false;
static std::vector<std::thread>     g_workers;

/* shrink default thread stack on Linux/glibc ≥2.34 */
#if defined(__linux__)
inline void shrink_default_stack()
{
    static bool done = false;
    if (done) return;
    done = true;
  #if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 34)
    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);           // 1 MiB
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
  #endif
}
#endif

/* worker loop */
static void worker_loop()
{
    for (;;) {
        std::unique_lock<std::mutex> ul(g_mtx);
        g_cv.wait(ul, [] { return g_tasks || g_stop; });
        if (g_stop) return;

        /* local shared_ptr keeps the vector alive while this thread works */
        auto tasks = g_tasks;
        ul.unlock();

        size_t idx = g_next.fetch_add(1);
        while (idx < tasks->size()) {
            try {
                save_slice((*tasks)[idx]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(g_mtx);
                g_errs.emplace_back(e.what());
            }

            if (g_remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(g_mtx);
                g_tasks.reset();         // clear global ptr; vector freed when last thread releases its copy
                g_cv.notify_all();
            }
            idx = g_next.fetch_add(1);
        }
    }
}

/* create pool once */
inline void ensure_pool(size_t requested)
{
    if (!g_workers.empty()) return;

#if defined(__linux__)
    shrink_default_stack();
#endif
    requested = std::max<size_t>(1, requested);

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
        g_workers.clear();
    });
}

} // namespace

/*==============================================================================
    MEX gateway
==============================================================================*/
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        /* argument checks (unchanged) */
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

        /* build tasks */
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        auto taskVec = std::make_shared<std::vector<SaveTask>>();
        taskVec->reserve(dim2);

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* c = mxGetCell(prhs[1], z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString p(c);
            taskVec->push_back({base, dim0, dim1, z, p.get(),
                                isXYZ, mxGetClassID(V), comp});
        }

        if (taskVec->empty()) return;

        /* schedule batch */
        const size_t poolSize =
            std::min<size_t>(std::thread::hardware_concurrency(), 32);
        ensure_pool(poolSize);

        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_tasks      = taskVec;
            g_next       = 0;
            g_remaining  = taskVec->size();
            g_errs.clear();
        }
        g_cv.notify_all();

        /* wait for completion; g_tasks reset by last worker */
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
