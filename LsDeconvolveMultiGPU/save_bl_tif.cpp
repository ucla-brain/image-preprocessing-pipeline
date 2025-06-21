/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 2-D / 3-D MATLAB arrays
  (one TIFF per slice, optional LZW / Deflate compression).

  Author   : Keivan Moradi  (with ChatGPT assistance)
  License  : GNU GPL v3   <https://www.gnu.org/licenses/>

  OVERVIEW
  --------
  • Accepts uint8 or uint16 input.
  • Accepts:
        – 2-D  matrix  [Y × X]          ➔ treated as one slice (Z=1)
        – 3-D  volume  [Y × X × Z]
  • Layout flag (orderFlag):
        false (default) → MATLAB native  [Y X Z]  ⇒ **transpose required**
        true            → User-permuted [X Y Z]  ⇒ already row-major
  • Per-call thread-pool (≥1, ≤HW threads, floor 8 when HW ≥ 8).
      – Lock-free atomic index dispenser (g_nextSlice).
      – Each thread owns a grow-only scratch buffer.
      – No inter-thread queues, mutexes, or condition variables in hot path.
  • 1 MiB pthread stacks on Linux/glibc ≥ 2.34.
  • Uses TIFFWriteRawStrip for “none”; TIFFWriteEncodedStrip otherwise.
  • Matches load_bl_tif.cpp for slice order and dimensions.

  MEMORY
  ------
  Extra RAM ≈ (#threads × largestSliceBytes).  No unbounded growth.

  USAGE
  -----
      save_bl_tif(volume3d, fileList, orderFlag, compression)

      volume3d    : uint8 | uint16, 2-D or 3-D
      fileList    : 1×Z cell array of output paths (char / string)
      orderFlag   : logical | uint32   (true → [X Y Z], false → [Y X Z])
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
#   include <pthread.h>
#   include <fcntl.h>   // posix_fadvise
#   include <unistd.h>
#endif

/* ------------------------------------------------------------------------- */
/*                    Small UTF-8 helper for MATLAB strings                  */
/* ------------------------------------------------------------------------- */
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

/* ------------------------------------------------------------------------- */
/*            Immutable description of one slice in the current batch        */
/* ------------------------------------------------------------------------- */
struct SliceTask {
    const uint8_t* base;     // pointer to entire volume
    mwSize         dimY;     // rows  (Y)
    mwSize         dimX;     // cols  (X)
    mwSize         zIndex;   // slice index
    std::string    outPath;  // destination TIFF file
    bool           isXYZ;    // true → already row-major
    mxClassID      classId;  // uint8 / uint16
    std::string    compression; // "none" | "lzw" | "deflate"
};

/* ------------------------------------------------------------------------- */
/*                             Global batch state                            */
/* ------------------------------------------------------------------------- */
namespace {
std::shared_ptr<const std::vector<SliceTask>> g_tasks;
std::atomic_size_t  g_nextSlice   {0};   // lock-free ticket dispenser
std::atomic_size_t  g_slicesLeft  {0};   // countdown to zero

std::mutex              g_waitMtx;
std::condition_variable g_waitCV;

std::mutex              g_errMtx;
std::vector<std::string> g_errors;

std::vector<std::thread> g_pool;
} // namespace

/* ------------------------------------------------------------------------- */
/*            Per-thread scratch buffer (grows, never shrinks)               */
/* ------------------------------------------------------------------------- */
static thread_local std::vector<uint8_t> tls_buf;

/* ------------------------------------------------------------------------- */
/*                Automatically notifies main thread when done               */
/* ------------------------------------------------------------------------- */
struct SliceDone {
    ~SliceDone() {
        if (g_slicesLeft.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk(g_waitMtx);
            g_waitCV.notify_one();
        }
    }
};

/* ------------------------------------------------------------------------- */
/*                     Worker thread  (compute + I/O)                        */
/* ------------------------------------------------------------------------- */
static void worker()
{
    try {
        for (;;) {
            /* ---------------------- claim next ticket ------------------ */
            std::size_t idx = g_nextSlice.fetch_add(1);
            if (idx >= g_tasks->size())
                break;                      // all slices processed

            const SliceTask& t = (*g_tasks)[idx];
            SliceDone        doneGuard;     // always decrements g_slicesLeft

            const std::size_t bpp   = (t.classId == mxUINT16_CLASS ? 2 : 1);
            const std::size_t nPix  = static_cast<std::size_t>(t.dimY) * t.dimX;
            const std::size_t nByte = nPix * bpp;

            if (tls_buf.size() < nByte) tls_buf.resize(nByte);
            uint8_t*       dst = tls_buf.data();
            const uint8_t* srcBase =
                t.base + static_cast<std::size_t>(t.zIndex) * nPix * bpp;

            /* ------------------ transpose / copy ----------------------- */
            if (!t.isXYZ) {                              /* Y-X-Z  → transpose */
                for (mwSize col = 0; col < t.dimX; ++col) {
                    const uint8_t* srcCol =
                        srcBase + static_cast<std::size_t>(col) * t.dimY * bpp;
                    for (mwSize row = 0; row < t.dimY; ++row) {
                        std::size_t di =
                            (static_cast<std::size_t>(row) * t.dimX + col) * bpp;
                        std::memcpy(dst + di,
                                    srcCol + static_cast<std::size_t>(row) * bpp,
                                    bpp);
                    }
                }
            } else {                                     /* X-Y-Z  → copy rows */
                const std::size_t rowBytes = t.dimX * bpp;
                for (mwSize row = 0; row < t.dimY; ++row) {
                    const uint8_t* srcRow =
                        srcBase + static_cast<std::size_t>(row) * t.dimX * bpp;
                    std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                                srcRow, rowBytes);
                }
            }

            /* ---------------------- write TIFF ------------------------- */
            TIFF* tif = TIFFOpen(t.outPath.c_str(), "w");
            if (!tif)
                throw std::runtime_error("Cannot open " + t.outPath);

            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  t.dimX);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, t.dimY);
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,  (bpp == 2 ? 16 : 8));

            uint16_t compTag = (t.compression == "none")   ? COMPRESSION_NONE   :
                               (t.compression == "lzw")    ? COMPRESSION_LZW    :
                                                             COMPRESSION_DEFLATE;
            TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, t.dimY);

            tsize_t wrote = (t.compression == "none")
                ? TIFFWriteRawStrip    (tif, 0, dst, nByte)
                : TIFFWriteEncodedStrip(tif, 0, dst, nByte);

            if (wrote < 0) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF write failed on " + t.outPath);
            }

#if defined(__linux__)
            int fd = TIFFFileno(tif);
            if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            TIFFClose(tif);
        }
    }
    catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(g_errMtx);
        g_errors.emplace_back(e.what());
        /* ensure main thread wakes even if an error stops slices early */
        if (g_slicesLeft.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk2(g_waitMtx);
            g_waitCV.notify_one();
        }
    }
}

/* ------------------------------------------------------------------------- */
/*             Linux: shrink default pthread stack to 1 MiB (optional)       */
/* ------------------------------------------------------------------------- */
#if defined(__linux__)
static void shrink_default_stack()
{
#   if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2,34)
    static bool done = false;
    if (done) return;  done = true;

    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
#   endif
}
#endif

/* ------------------------------------------------------------------------- */
static void spawn_pool(std::size_t nThreads)
{
#if defined(__linux__)
    shrink_default_stack();
#endif
    g_pool.clear();
    g_pool.reserve(nThreads);
    for (std::size_t i = 0; i < nThreads; ++i)
        g_pool.emplace_back(worker);
}

/* ======================================================================= */
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        /* ---------------- argument checks ----------------------------- */
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
              "Usage: save_bl_tif(volume, fileList, orderFlag, compression)");

        const mxArray* vol = prhs[0];
        if (!mxIsUint8(vol) && !mxIsUint16(vol))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        const mwSize* dims = mxGetDimensions(vol);
        mwSize nd = mxGetNumberOfDimensions(vol);

        mwSize dimY, dimX, dimZ;
        if (nd == 2) { dimY = dims[0]; dimX = dims[1]; dimZ = 1; }
        else if (nd == 3){ dimY = dims[0]; dimX = dims[1]; dimZ = dims[2]; }
        else mexErrMsgIdAndTxt("save_bl_tif:Input",
                               "Volume must be 2-D or 3-D");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dimZ)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
              *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString compStr(prhs[3]);
        std::string compression(compStr.get());
        if (compression != "none" && compression != "lzw" && compression != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        /* ---------------- build task list ----------------------------- */
        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(vol));
        auto tasks          = std::make_shared<std::vector<SliceTask>>();
        tasks->reserve(dimZ);

        for (mwSize z = 0; z < dimZ; ++z) {
            const mxArray* c = mxGetCell(prhs[1], z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString p(c);
            tasks->push_back(
                { base, dimY, dimX, z, p.get(),
                  isXYZ, mxGetClassID(vol), compression });
        }
        if (tasks->empty()) return;

        /* ---------------- publish batch & counters -------------------- */
        g_tasks       = std::move(tasks);
        g_nextSlice   = 0;
        g_slicesLeft  = g_tasks->size();
        g_errors.clear();

        /* ---------------- spawn pool (fresh every call) --------------- */
        std::size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 8;
        std::size_t nThreads = std::min(hw, g_tasks->size());
        if (hw >= 8) nThreads = std::max<std::size_t>(8, nThreads);
        nThreads = std::max<std::size_t>(1, nThreads);

        spawn_pool(nThreads);

        /* ---------------- wait for completion ------------------------- */
        {
            std::unique_lock<std::mutex> lk(g_waitMtx);
            g_waitCV.wait(lk, []{ return g_slicesLeft.load() == 0; });
        }
        for (auto& t : g_pool) if (t.joinable()) t.join();
        g_pool.clear();

        /* ---------------- propagate errors ---------------------------- */
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
