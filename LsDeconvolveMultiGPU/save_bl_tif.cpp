/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 2-D / 3-D MATLAB arrays
  (one TIFF per slice, optional LZW / Deflate compression).

  Author   : Keivan Moradi (with ChatGPT assistance)
  License  : GNU GPL-3 <https://www.gnu.org/licenses/>

  HIGHLIGHTS
  ----------
  • uint8 / uint16 input, 2-D promoted to [Y X 1].
  • Thread pool per MATLAB session
        – N “compute” workers (copy / transpose)
        – 1 writer (TIFF output)
    Bounded queue (16 jobs) ⇒ compute/IO overlap, bounded RAM.
  • 1 MiB pthread stacks (glibc ≥ 2.34); posix_fadvise(DONTNEED) per slice.
  • TIFFWriteRawStrip for "none", TIFFWriteEncodedStrip otherwise.
  • Pool size ≤ min(Z, hw threads) and ≥ 8 where possible.
==============================================================================*/

#include "mex.h"
#include "matrix.h"
#include "tiffio.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
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

/*==============================================================================
  UTF-8 helper for MATLAB strings
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
  Per-slice task (input → compute)
==============================================================================*/
struct SliceTask {
    const uint8_t* base;       // pointer to whole volume
    mwSize         dimY;       // MATLAB rows  (Y)
    mwSize         dimX;       // MATLAB cols  (X)
    mwSize         zIndex;     // slice index
    std::string    outPath;    // destination TIFF
    bool           isXYZ;      // true = [X Y Z] layout
    mxClassID      classId;    // uint8 / uint16
    std::string    compression;
};

/*==============================================================================
  Job handed to the writer (owns pixel buffer)
==============================================================================*/
struct WriteJob {
    std::string          outPath;
    std::vector<uint8_t> pixels;       // already transposed / packed
    mwSize               rows, cols;   // TIFF dimensions
    mxClassID            classId;
    std::string          compression;
};

/*==============================================================================
  Global concurrency state
==============================================================================*/
namespace {

constexpr std::size_t                    kQueueCap = 16;     // bounded FIFO
std::deque<WriteJob>                     g_queue;
std::mutex                               g_qMtx;
std::condition_variable                  g_qNotEmpty;
std::condition_variable                  g_qNotFull;

std::shared_ptr<const std::vector<SliceTask>> g_tasks;
std::atomic_size_t                       g_nextIdx{0};       // slice dispatch
std::atomic_size_t                       g_leftToProduce{0}; // slices yet to produce

std::mutex                               g_errMtx;
std::vector<std::string>                 g_errors;

std::mutex                               g_poolMtx;
std::vector<std::thread>                 g_threads;
bool                                     g_poolAlive = false;

} // namespace

/*==============================================================================
  Writer thread – TIFF output
==============================================================================*/
static void writer_thread()
{
    try {
        for (;;) {
            /* ---------- wait for an available job or completion ---------- */
            std::unique_lock<std::mutex> ul(g_qMtx);
            g_qNotEmpty.wait(ul, []{
                return !g_queue.empty() || g_leftToProduce.load() == 0;
            });

            if (g_queue.empty() && g_leftToProduce.load() == 0) {
                ul.unlock();
                g_qNotEmpty.notify_all();              // wake MATLAB thread
                break;                                 // finished
            }

            WriteJob job = std::move(g_queue.front());
            g_queue.pop_front();
            g_qNotFull.notify_one();
            ul.unlock();

            /* --------------------- write a TIFF ------------------------- */
            const std::size_t bpp = (job.classId == mxUINT16_CLASS ? 2 : 1);

            TIFF* tif = TIFFOpen(job.outPath.c_str(), "w");
            if (!tif)
                throw std::runtime_error("Cannot open " + job.outPath);

            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  job.cols);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, job.rows);
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,  bpp == 2 ? 16 : 8);

            const bool raw = (job.compression == "none");
            const uint16_t compTag = raw ? COMPRESSION_NONE :
                                  (job.compression == "lzw") ? COMPRESSION_LZW :
                                                               COMPRESSION_DEFLATE;
            TIFFSetField(tif, TIFFTAG_COMPRESSION,  compTag);
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, job.rows);

            const tsize_t wrote = raw
                ? TIFFWriteRawStrip    (tif, 0, job.pixels.data(), job.pixels.size())
                : TIFFWriteEncodedStrip(tif, 0, job.pixels.data(), job.pixels.size());

            if (wrote < 0) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF write failed on " + job.outPath);
            }

#if defined(__linux__)
            if (int fd = TIFFFileno(tif); fd != -1)
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            TIFFClose(tif);
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(g_errMtx);
        g_errors.emplace_back(e.what());
        g_qNotEmpty.notify_all();                       // unblock MATLAB
    }
}

/*==============================================================================
  Compute worker – copy / transpose then queue
==============================================================================*/
static void compute_thread()
{
    for (;;) {
        /* -------------- claim next slice index -------------------------- */
        size_t idx = g_nextIdx.fetch_add(1);
        if (idx >= g_tasks->size())
            break;

        const SliceTask& t = (*g_tasks)[idx];

        /* ---------- build WriteJob buffer ------------------------------- */
        const std::size_t bpp      = (t.classId == mxUINT16_CLASS ? 2 : 1);
        const mwSize      rowsOut  = t.dimY;
        const mwSize      colsOut  = t.dimX;
        const std::size_t pixels   = static_cast<std::size_t>(rowsOut) * colsOut;
        const std::size_t bytes    = pixels * bpp;

        WriteJob job;
        job.outPath     = t.outPath;
        job.rows        = rowsOut;
        job.cols        = colsOut;
        job.classId     = t.classId;
        job.compression = t.compression;
        job.pixels.resize(bytes);

        uint8_t*       dst = job.pixels.data();
        const uint8_t* srcSlice =
            t.base + static_cast<std::size_t>(t.zIndex) * pixels * bpp;

        /* ---------- copy / transpose ------------------------------------ */
        if (!t.isXYZ) {                                   // [Y X Z] → transpose
            for (mwSize col = 0; col < colsOut; ++col) {
                const uint8_t* srcCol =
                    srcSlice + static_cast<std::size_t>(col) * rowsOut * bpp;
                for (mwSize row = 0; row < rowsOut; ++row) {
                    std::size_t dstIdx =
                        (static_cast<std::size_t>(row) * colsOut + col) * bpp;
                    std::memcpy(dst + dstIdx,
                                srcCol + static_cast<std::size_t>(row) * bpp,
                                bpp);
                }
            }
        } else {                                          // [X Y Z] → fast path
            const std::size_t rowBytes = colsOut * bpp;
            for (mwSize row = 0; row < rowsOut; ++row) {
                const uint8_t* srcRow =
                    srcSlice + static_cast<std::size_t>(row) * rowsOut * bpp;
                std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                            srcRow, rowBytes);
            }
        }

        /* ---------- enqueue job & signal writer ------------------------- */
        {
            std::unique_lock<std::mutex> ql(g_qMtx);
            g_qNotFull.wait(ql, []{ return g_queue.size() < kQueueCap; });
            g_queue.emplace_back(std::move(job));

            std::size_t prev = g_leftToProduce.fetch_sub(1);
            if (prev == 1)               // last slice just produced
                g_qNotEmpty.notify_all();
            else
                g_qNotEmpty.notify_one();
        }
    }
}

/*==============================================================================
  Linux: shrink default pthread stack to 1 MiB (glibc ≥ 2.34)
==============================================================================*/
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

/*==============================================================================
  Thread-pool bootstrap (run once per MATLAB session)
==============================================================================*/
static void ensure_pool(std::size_t nSlices)
{
    std::lock_guard<std::mutex> lk(g_poolMtx);
    if (g_poolAlive) return;

#if defined(__linux__)
    shrink_default_stack();
#endif
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;

    std::size_t nCompute = std::min(nSlices, hw);
    if (hw >= 8) nCompute = std::max<std::size_t>(8, nCompute);
    nCompute = std::max<std::size_t>(1, nCompute);

    g_threads.reserve(nCompute + 1);
    g_threads.emplace_back(writer_thread);
    for (std::size_t i = 0; i < nCompute; ++i)
        g_threads.emplace_back(compute_thread);

    g_poolAlive = true;

    mexAtExit(+[] {
        {
            std::lock_guard<std::mutex> lk(g_qMtx);
            g_leftToProduce = 0;
            g_queue.clear();
        }
        g_qNotEmpty.notify_all();
        g_qNotFull.notify_all();
        for (auto& t : g_threads)
            if (t.joinable()) t.join();
    });
}

/*==============================================================================
  MEX gateway
==============================================================================*/
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
              "Usage: save_bl_tif(volume, fileList, orderFlag, compression)");

        /* ---------------- validate input ------------------------------- */
        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        const mwSize* d = mxGetDimensions(V);
        mwSize nd = mxGetNumberOfDimensions(V);

        mwSize dimY, dimX, dimZ;
        if (nd == 2)      { dimY = d[0]; dimX = d[1]; dimZ = 1; }
        else if (nd == 3) { dimY = d[0]; dimX = d[1]; dimZ = d[2]; }
        else
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Volume must be 2-D or 3-D.");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dimZ)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        const bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                          ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
                           *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString compStr(prhs[3]);
        std::string  compression(compStr.get());
        if (compression != "none" && compression != "lzw" && compression != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        /* ---------------- build slice-task vector ---------------------- */
        auto tasks = std::make_shared<std::vector<SliceTask>>();
        tasks->reserve(dimZ);

        const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(V));

        for (mwSize z = 0; z < dimZ; ++z) {
            const mxArray* c = mxGetCell(prhs[1], z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString path(c);
            tasks->push_back(SliceTask{ basePtr, dimY, dimX, z,
                                        path.get(), isXYZ,
                                        mxGetClassID(V), compression });
        }

        if (tasks->empty()) return;

        /* ---------------- publish tasks & counters --------------------- */
        {
            std::lock_guard<std::mutex> lk(g_qMtx);
            g_tasks           = std::move(tasks);
            g_nextIdx         = 0;
            g_leftToProduce   = g_tasks->size();
        }

        /* ---------------- start / wake thread-pool --------------------- */
        ensure_pool(g_tasks->size());

        /* ---------------- wait for completion -------------------------- */
        {
            std::unique_lock<std::mutex> ul(g_qMtx);
            g_qNotEmpty.wait(ul, []{
                return g_leftToProduce.load() == 0 && g_queue.empty();
            });
        }

        /* ---------------- propagate worker errors ---------------------- */
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
