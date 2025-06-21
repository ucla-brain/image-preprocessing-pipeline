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
      – Fast path for [Y X Z] avoids per-pixel multiplications.
      – Matches `load_bl_tif.cpp` for slice order and dimensions.
      – Pool size clamped to min(Z-slices, HW threads); excess threads stay idle.

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
# include <pthread.h>
# include <fcntl.h>
# include <unistd.h>
#endif

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

struct SaveTask {
    const uint8_t* base;
    mwSize         dim0;   // Y
    mwSize         dim1;   // X
    mwSize         z;      // slice index
    std::string    path;
    bool           isXYZ;
    mxClassID      classId;
    std::string    comp;
};

static thread_local std::vector<uint8_t> scratch;

static void save_slice(const SaveTask& t)
{
    const size_t bytesPerPixel = (t.classId == mxUINT16_CLASS ? 2 : 1);

    const mwSize srcRows    = t.isXYZ ? t.dim1 : t.dim0;
    const mwSize srcCols    = t.isXYZ ? t.dim0 : t.dim1;
    const size_t sliceIndex = static_cast<size_t>(t.z);
    const size_t pixelsPerSlice = static_cast<size_t>(t.dim0) * t.dim1;
    const size_t bytesPerSlice  = pixelsPerSlice * bytesPerPixel;

    if (scratch.capacity() < bytesPerSlice) {
        scratch.reserve(bytesPerSlice);
        scratch.resize(bytesPerSlice);
    }
    uint8_t* dstBuffer = scratch.data();

    if (!t.isXYZ) {                     /* --- MATLAB default [Y X Z] --- */
        for (mwSize col = 0; col < srcCols; ++col) {
            const uint8_t* srcColumn =
                t.base + (sliceIndex * pixelsPerSlice +
                          static_cast<size_t>(col) * t.dim0) * bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row) {
                size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * bytesPerPixel;
                const uint8_t* src = srcColumn + static_cast<size_t>(row) * bytesPerPixel;
                if (bytesPerPixel == 1)
                    dstBuffer[dstIdx] = *src;
                else
                    std::memcpy(dstBuffer + dstIdx, src, 2);
            }
        }
    }
    else {
        const size_t srcRowBytes = t.dim0 * bytesPerPixel;
        const size_t dstRowBytes = srcCols * bytesPerPixel;
        const size_t baseBytes   = sliceIndex * pixelsPerSlice * bytesPerPixel;

        for (mwSize row = 0; row < srcRows; ++row) {
            const uint8_t* srcRow =
                t.base + baseBytes + static_cast<size_t>(row) * t.dim0 * bytesPerPixel;
            std::memcpy(dstBuffer + static_cast<size_t>(row) * dstRowBytes,
                        srcRow, dstRowBytes);
        }
    }

    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,  bytesPerPixel == 2 ? 16 : 8);

    const bool isRaw = (t.comp == "none");
    const uint16_t compTag = isRaw ? COMPRESSION_NONE :
                              (t.comp == "lzw") ? COMPRESSION_LZW : COMPRESSION_DEFLATE;

    TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    const tsize_t wrote = isRaw
        ? TIFFWriteRawStrip    (tif, 0, dstBuffer, bytesPerSlice)
        : TIFFWriteEncodedStrip(tif, 0, dstBuffer, bytesPerSlice);

    if (wrote < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF write failed on " + t.path);
    }

#if defined(__linux__)
    int fd = TIFFFileno(tif);
    if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    TIFFClose(tif);
}

namespace {
static std::shared_ptr<const std::vector<SaveTask>> g_tasks;
static std::atomic_size_t g_next{0};
static std::atomic_size_t g_remaining{0};
static std::vector<std::string> g_errs;
static std::mutex g_mtx;
static std::condition_variable g_cv;
static bool g_stop = false;
static std::vector<std::thread> g_workers;

#if defined(__linux__)
inline void shrink_stack()
{
    static bool done = false;
    if (done) return; done = true;
# if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2,34)
    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
# endif
}
#endif

static void worker_loop()
{
    for (;;) {
        std::unique_lock<std::mutex> lock(g_mtx);
        g_cv.wait(lock, []{ return g_tasks || g_stop; });
        if (g_stop) return;
        auto tasks = g_tasks;
        lock.unlock();

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

inline void ensure_pool(size_t nSlices)
{
    if (!g_workers.empty()) return;
#if defined(__linux__)
    shrink_stack();
#endif
    size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;
    size_t nThreads = std::min(nSlices, hw);
    if (hw >= 8) nThreads = std::max<size_t>(8, nThreads);
    if (nThreads == 0) nThreads = 1;

    g_workers.reserve(nThreads);
    for (size_t i = 0; i < nThreads; ++i)
        g_workers.emplace_back(worker_loop);

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
} // namespace

void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        const mwSize* dims = mxGetDimensions(V);
        mwSize nd = mxGetNumberOfDimensions(V);

        mwSize dim0, dim1, dim2;
        if (nd == 2) { dim0 = dims[0]; dim1 = dims[1]; dim2 = 1; }
        else if (nd == 3) { dim0 = dims[0]; dim1 = dims[1]; dim2 = dims[2]; }
        else mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be 2-D or 3-D.");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        const bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
             *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString cs(prhs[3]);
        std::string comp(cs.get());
        if (comp != "none" && comp != "lzw" && comp != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        auto tasks = std::make_shared<std::vector<SaveTask>>();
        tasks->reserve(dim2);

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* cell = mxGetCell(prhs[1], z);
            if (!mxIsChar(cell))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");
            MatlabString path(cell);
            tasks->push_back({ base, dim0, dim1, z, path.get(),
                               isXYZ, mxGetClassID(V), comp });
        }

        if (tasks->empty()) return;
        ensure_pool(tasks->size());

        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_tasks     = tasks;
            g_next      = 0;
            g_remaining = tasks->size();
            g_errs.clear();
        }
        g_cv.notify_all();

        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv.wait(lk, [] { return !g_tasks; });
        }

        if (!g_errs.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : g_errs) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
