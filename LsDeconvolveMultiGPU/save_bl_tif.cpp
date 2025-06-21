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
      - Accepts 2-D (promoted to 3-D [Y X 1]), or true 3-D, uint8 | uint16 data.
      - Thread pool:
            – N compute workers   (transpose / copy)
            – 1 I/O writer thread (TIFF output)
        Queue depth 16  ⇒  compute + I/O overlap without unbounded RAM growth.
      - Uses TIFFWriteRawStrip for "none" compression; TIFFWriteEncodedStrip
        otherwise.
      - Linux:  posix_fadvise DONTNEED after every write; 1 MiB pthread stacks.

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
  RAII helper to turn MATLAB char/ string into UTF-8 std::string
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
  Task description — one per Z-slice (input phase)
==============================================================================*/
struct SliceTask {
    const uint8_t* base;          // pointer to whole volume (MATLAB memory)
    mwSize         dimY;          // rows   in MATLAB storage (Y)
    mwSize         dimX;          // cols   in MATLAB storage (X)
    mwSize         zIndex;        // slice index 0-based
    bool           isXYZlayout;   // true => [X Y Z]  (already row-major)
    mxClassID      classId;       // uint8 / uint16
    std::string    outPath;
    std::string    compression;   // "none" | "lzw" | "deflate"
};

/*==============================================================================
  Write job — owns pixel buffer for one slice (output phase)
==============================================================================*/
struct WriteJob {
    std::string           outPath;
    std::vector<uint8_t>  pixels;     // already transposed / packed
    mwSize                rows, cols; // final TIFF image size
    mxClassID             classId;
    std::string           compression;
};

/*==============================================================================
  Global state — producer / consumer queue and thread pool
==============================================================================*/
namespace {

constexpr std::size_t              kQueueCapacity = 16;   // bounded queue
std::deque<WriteJob>               g_queue;               // job FIFO
std::mutex                         g_queueMtx;
std::condition_variable            g_queueNotEmpty;
std::condition_variable            g_queueNotFull;

std::shared_ptr<const std::vector<SliceTask>> g_tasks;    // all slice tasks
std::atomic_size_t               g_nextTask{0};           // index dispatch
std::atomic_size_t               g_computeRemaining{0};   // producers active

/* error collection */
std::mutex                       g_errMtx;
std::vector<std::string>         g_errors;

/* pool control */
std::mutex                       g_poolMtx;
std::vector<std::thread>         g_threads;               // workers + writer
bool                             g_stopPool = false;

} // anonymous namespace

/*==============================================================================
  TIFF writer (consumer)
==============================================================================*/
static void io_writer()
{
    try {
        for (;;) {
            /* --- wait for a job ----------------------------------------- */
            std::unique_lock<std::mutex> ql(g_queueMtx);
            g_queueNotEmpty.wait(ql, []{
                return !g_queue.empty() ||
                       (g_computeRemaining.load() == 0);    // no more producers
            });

            if (g_queue.empty() && g_computeRemaining.load() == 0)
                break;                                     // all done

            WriteJob job = std::move(g_queue.front());
            g_queue.pop_front();
            g_queueNotFull.notify_one();
            ql.unlock();

            /* --- write TIFF --------------------------------------------- */
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
            int fd = TIFFFileno(tif);
            if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            TIFFClose(tif);
        }
    }
    catch (const std::exception& e) {
        std::lock_guard<std::mutex> el(g_errMtx);
        g_errors.emplace_back(e.what());
    }
}

/*==============================================================================
  Compute worker (producer) — transpose / copy, then queue
==============================================================================*/
static void compute_worker()
{
    for (;;) {
        /* fetch next slice index ------------------------------------------ */
        size_t idx = g_nextTask.fetch_add(1);
        if (idx >= g_tasks->size())
            break;

        const SliceTask& task = (*g_tasks)[idx];

        /* figure out final image size ------------------------------------- */
        const std::size_t bpp = (task.classId == mxUINT16_CLASS ? 2 : 1);
        const mwSize rowsFinal = task.isXYZlayout ? task.dim1 : task.dim0;
        const mwSize colsFinal = task.isXYZlayout ? task.dim0 : task.dim1;
        const std::size_t pixelsPerSlice = static_cast<std::size_t>(task.dim0) * task.dim1;
        const std::size_t bytesPerSlice  = pixelsPerSlice * bpp;

        WriteJob job;
        job.outPath     = task.outPath;
        job.rows        = rowsFinal;
        job.cols        = colsFinal;
        job.classId     = task.classId;
        job.compression = task.compression;
        job.pixels.resize(bytesPerSlice);

        uint8_t* dst = job.pixels.data();
        const uint8_t* srcVolumeBase = task.base +
            static_cast<std::size_t>(task.zIndex) * pixelsPerSlice * bpp;

        /* ------------------ transpose / copy ----------------------------- */
        if (!task.isXYZlayout) {   /* MATLAB default [Y X Z] (column-major) */
            for (mwSize col = 0; col < colsFinal; ++col) {
                const uint8_t* srcColumn =
                    srcVolumeBase + static_cast<std::size_t>(col) * task.dim0 * bpp;

                for (mwSize row = 0; row < rowsFinal; ++row) {
                    std::size_t dstIdx = (static_cast<std::size_t>(row) * colsFinal + col) * bpp;
                    std::memcpy(dst + dstIdx,
                                srcColumn + static_cast<std::size_t>(row) * bpp,
                                bpp);
                }
            }
        } else {                   /* [X Y Z] already row-major */
            const std::size_t rowBytes = colsFinal * bpp;
            for (mwSize row = 0; row < rowsFinal; ++row) {
                const uint8_t* srcRow =
                    srcVolumeBase + static_cast<std::size_t>(row) * task.dim0 * bpp;
                std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                            srcRow, rowBytes);
            }
        }

        /* ------------------ enqueue -------------------------------------- */
        std::unique_lock<std::mutex> ql(g_queueMtx);
        g_queueNotFull.wait(ql, []{ return g_queue.size() < kQueueCapacity; });
        g_queue.emplace_back(std::move(job));
        ql.unlock();
        g_queueNotEmpty.notify_one();
    }

    /* producer done */
    if (g_computeRemaining.fetch_sub(1) == 1)
        g_queueNotEmpty.notify_one();   // wake writer for shutdown
}

/*==============================================================================
  Helper: shrink default pthread stack on glibc ≥ 2.34 (Linux only)
==============================================================================*/
#if defined(__linux__)
inline void shrink_default_stack()
{
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2,34)
    static bool done = false;
    if (done) return;
    done = true;

    pthread_attr_t a;
    if (!pthread_attr_init(&a)) {
        pthread_attr_setstacksize(&a, 1 << 20);   // 1 MiB
        pthread_setattr_default_np(&a);
        pthread_attr_destroy(&a);
    }
#endif
}
#endif

/*==============================================================================
  Create pool (compute + writer) exactly once per MATLAB session
==============================================================================*/
inline void ensure_pool(std::size_t numSlices)
{
    std::lock_guard<std::mutex> lock(g_poolMtx);
    if (!g_threads.empty()) return;        // already created

#if defined(__linux__)
    shrink_default_stack();
#endif
    std::size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 8;

    std::size_t computeThreads = std::min(numSlices, hw);
    if (hw >= 8) computeThreads = std::max<std::size_t>(8, computeThreads);
    computeThreads = std::max<std::size_t>(1, computeThreads);

    g_computeRemaining = numSlices;

    /* spawn threads: one writer + N compute */
    g_threads.reserve(computeThreads + 1);
    g_threads.emplace_back(io_writer);
    for (std::size_t i = 0; i < computeThreads; ++i)
        g_threads.emplace_back(compute_worker);

    /* join on mexClear / ctrl-C */
    mexAtExit(+[] {
        {
            std::lock_guard<std::mutex> lk(g_poolMtx);
            g_stopPool = true;
        }
        g_queueNotEmpty.notify_all();
        g_queueNotFull.notify_all();
        for (auto& th : g_threads)
            if (th.joinable()) th.join();
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

        const mxArray* vol = prhs[0];
        if (!mxIsUint8(vol) && !mxIsUint16(vol))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        /* ---- handle 2-D or 3-D input ---------------------------------- */
        const mwSize* dims = mxGetDimensions(vol);
        mwSize nd          = mxGetNumberOfDimensions(vol);

        mwSize dimY, dimX, dimZ;
        if (nd == 2) {
            dimY = dims[0]; dimX = dims[1]; dimZ = 1;
        } else if (nd == 3) {
            dimY = dims[0]; dimX = dims[1]; dimZ = dims[2];
        } else {
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Volume must be 2-D or 3-D.");
        }

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

        /* ---- build slice task list ------------------------------------ */
        const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(vol));
        auto taskVec           = std::make_shared<std::vector<SliceTask>>();
        taskVec->reserve(dimZ);

        for (mwSize z = 0; z < dimZ; ++z) {
            const mxArray* c = mxGetCell(prhs[1], z);
            if (!mxIsChar(c))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");

            MatlabString path(c);
            taskVec->push_back({basePtr,
                                dimY, dimX, z,
                                path.get(),
                                isXYZ,
                                mxGetClassID(vol),
                                compression});
        }

        if (taskVec->empty()) return;   // nothing to do

        /* ---- hand over to pool ---------------------------------------- */
        ensure_pool(taskVec->size());

        {
            std::lock_guard<std::mutex> lk(g_poolMtx);
            g_tasks        = std::move(taskVec);
            g_nextTask     = 0;
        }

        /* wait until writer empties queue & finishes */
        {
            std::unique_lock<std::mutex> lk(g_queueMtx);
            g_queueNotEmpty.wait(lk, []{
                return g_computeRemaining.load() == 0 && g_queue.empty();
            });
        }

        /* check for errors thrown by any thread ------------------------- */
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
