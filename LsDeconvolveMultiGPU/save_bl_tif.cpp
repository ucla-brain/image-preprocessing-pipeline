/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 2-D / 3-D MATLAB arrays
  (one TIFF per slice, optional LZW / Deflate compression).

  Author : Keivan Moradi  (with ChatGPT assistance)
  License: GNU GPL v3   <https://www.gnu.org/licenses/>

  KEY FEATURES
  ------------
  • uint8 | uint16 input, 2-D promoted to [Y X 1]
  • Reusable thread pool:
        one worker per HW thread (≥ 8) – each worker both transposes & writes
  • Zero mutexes on the hot path – slice indices are handed out with
        g_nextSlice.fetch_add(1)
  • 1 MiB pthread stacks (glibc ≥ 2.34); posix_fadvise(DONTNEED) per slice
  • Matches load_bl_tif.cpp for slice order and dimensions
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
# include <pthread.h>
# include <fcntl.h>   // posix_fadvise
# include <unistd.h>
#endif

/*=============================================================================
  Small RAII helper: convert MATLAB char / string → UTF-8
=============================================================================*/
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

/*=============================================================================
  Slice task (immutable – shared by all workers)
=============================================================================*/
struct SliceTask {
    const uint8_t* base;   // pointer to full volume
    mwSize         dimY;
    mwSize         dimX;
    mwSize         z;      // slice index
    std::string    outPath;
    bool           isXYZ;  // true  → input already row-major
    mxClassID      classId;
    std::string    compression;   // "none" | "lzw" | "deflate"
};

/*=============================================================================
  Global pool state
=============================================================================*/
namespace {

std::shared_ptr<const std::vector<SliceTask>> g_tasks;     // current batch
std::atomic_size_t        g_nextSlice{0};  // atomic index dispenser
std::atomic_size_t        g_activeSlices{0};

std::mutex                g_finishMtx;
std::condition_variable   g_finishCV;

std::mutex                g_errorMtx;
std::vector<std::string>  g_errors;

/* thread pool (created once per MATLAB session) */
std::vector<std::thread>  g_workers;

} // anonymous namespace

/*=============================================================================
  Per-thread scratch buffer (grow-only)
=============================================================================*/
static thread_local std::vector<uint8_t> tls_scratch;

/*=============================================================================
  Worker – fetch / transpose / write
=============================================================================*/
static void worker_loop()
{
    try {
        for (;;) {
            std::size_t idx = g_nextSlice.fetch_add(1);
            if (idx >= g_tasks->size()) break;

            const SliceTask& t = (*g_tasks)[idx];
            const std::size_t bpp = (t.classId == mxUINT16_CLASS ? 2 : 1);
            const std::size_t pixels =
                static_cast<std::size_t>(t.dimY) * t.dimX;
            const std::size_t bytes = pixels * bpp;

            if (tls_scratch.capacity() < bytes) {
                tls_scratch.reserve(bytes);
                tls_scratch.resize(bytes);
            }
            uint8_t* dst = tls_scratch.data();

            const uint8_t* srcBase =
                t.base + static_cast<std::size_t>(t.z) * pixels * bpp;

            /* -------- transpose / copy -------------------------------- */
            if (!t.isXYZ) {                          // MATLAB default [Y X Z]
                for (mwSize col = 0; col < t.dimX; ++col) {
                    const uint8_t* srcCol =
                        srcBase + static_cast<std::size_t>(col) * t.dimY * bpp;
                    for (mwSize row = 0; row < t.dimY; ++row) {
                        std::size_t dstIdx =
                            (static_cast<std::size_t>(row) * t.dimX + col) * bpp;
                        std::memcpy(dst + dstIdx,
                                    srcCol + static_cast<std::size_t>(row) * bpp,
                                    bpp);
                    }
                }
            } else {                                 // already row-major
                const std::size_t rowBytes = t.dimX * bpp;
                for (mwSize row = 0; row < t.dimY; ++row) {
                    const uint8_t* srcRow =
                        srcBase + static_cast<std::size_t>(row) * t.dimY * bpp;
                    std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                                srcRow, rowBytes);
                }
            }

            /* -------- write TIFF -------------------------------------- */
            TIFF* tif = TIFFOpen(t.outPath.c_str(), "w");
            if (!tif)
                throw std::runtime_error("Cannot open " + t.outPath);

            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  t.dimX);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, t.dimY);
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,  bpp == 2 ? 16 : 8);

            const bool raw = (t.compression == "none");
            const uint16_t compTag = raw ? COMPRESSION_NONE :
                                   (t.compression == "lzw") ? COMPRESSION_LZW :
                                                              COMPRESSION_DEFLATE;
            TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, t.dimY);

            const tsize_t wrote = raw
                ? TIFFWriteRawStrip    (tif, 0, dst, bytes)
                : TIFFWriteEncodedStrip(tif, 0, dst, bytes);

            if (wrote < 0) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF write failed on " + t.outPath);
            }

#if defined(__linux__)
            int fd = TIFFFileno(tif);
            if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            TIFFClose(tif);

            /* one slice finished */
            if (g_activeSlices.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(g_finishMtx);
                g_finishCV.notify_one();
            }
        }
    }
    catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(g_errorMtx);
        g_errors.emplace_back(e.what());
        /* wake MATLAB even on error */
        std::lock_guard<std::mutex> lk2(g_finishMtx);
        g_finishCV.notify_one();
    }
}

/*=============================================================================
  Linux: shrink default pthread stack to 1 MiB on glibc ≥ 2.34
=============================================================================*/
#if defined(__linux__)
static void shrink_default_stack()
{
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2,34)
    static bool done = false;
    if (done) return;
    done = true;

    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
#endif
}
#endif

/*=============================================================================
  Initialise pool (once per MATLAB session)
=============================================================================*/
static void ensure_pool(std::size_t numSlices)
{
    if (!g_workers.empty()) return;   // already created

#if defined(__linux__)
    shrink_default_stack();
#endif
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;
    std::size_t nThreads = std::min(hw, numSlices);
    if (hw >= 8) nThreads = std::max<std::size_t>(8, nThreads);
    nThreads = std::max<std::size_t>(1, nThreads);

    g_workers.reserve(nThreads);
    try {
        for (std::size_t i = 0; i < nThreads; ++i)
            g_workers.emplace_back(worker_loop);
    } catch (...) {
        for (auto& t : g_workers)
            if (t.joinable()) t.join();
        throw;
    }

    mexAtExit(+[] {
        for (auto& t : g_workers)
            if (t.joinable()) t.join();
    });
}

/*=============================================================================
  MEX gateway
=============================================================================*/
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        /* ------------ validate inputs ------------------------------- */
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
              "Usage: save_bl_tif(volume, fileList, orderFlag, compression)");

        const mxArray* vol = prhs[0];
        if (!mxIsUint8(vol) && !mxIsUint16(vol))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Volume must be uint8 or uint16");

        const mwSize* dims = mxGetDimensions(vol);
        mwSize nd = mxGetNumberOfDimensions(vol);

        mwSize dimY, dimX, dimZ;
        if (nd == 2) { dimY = dims[0]; dimX = dims[1]; dimZ = 1; }
        else if (nd == 3) { dimY = dims[0]; dimX = dims[1]; dimZ = dims[2]; }
        else mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be 2-D or 3-D");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dimZ)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        const bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
             *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString compStr(prhs[3]);
        std::string  compression(compStr.get());
        if (compression != "none" && compression != "lzw" && compression != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        /* ------------ build task vector ----------------------------- */
        const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(vol));
        auto taskVec = std::make_shared<std::vector<SliceTask>>();
        taskVec->reserve(dimZ);

        for (mwSize z = 0; z < dimZ; ++z) {
            const mxArray* cell = mxGetCell(prhs[1], z);
            if (!mxIsChar(cell))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString p(cell);
            taskVec->push_back(
                { basePtr, dimY, dimX, z, p.get(),
                  isXYZ, mxGetClassID(vol), compression });
        }

        if (taskVec->empty()) return;          // nothing to do

        /* ------------ hand to pool & wait --------------------------- */
        g_tasks          = std::move(taskVec);
        g_nextSlice      = 0;
        g_activeSlices   = g_tasks->size();

        ensure_pool(g_tasks->size());

        /* wait for completion */
        {
            std::unique_lock<std::mutex> lk(g_finishMtx);
            g_finishCV.wait(lk, []{ return g_activeSlices.load() == 0; });
        }

        /* propagate errors from worker threads ----------------------- */
        if (!g_errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : g_errors) msg += "  - " + e + '\n';
            g_errors.clear();
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
