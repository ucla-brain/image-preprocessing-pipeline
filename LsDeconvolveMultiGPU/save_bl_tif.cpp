/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  Version      : 2025-06-21 (persistent-pool, thread-reuse, bug-fixed)
  Author       : Keivan Moradi  (with ChatGPT-4o assistance)
  License      : GNU GPL v3   <https://www.gnu.org/licenses/>
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
# include <fcntl.h>
# include <unistd.h>
#endif

/* ───────────────────────────────── TASK DESCRIPTION ─────────────────────── */
struct SaveTask {
    const uint8_t* base;          // start of whole volume
    size_t         offset_bytes;  // start of this slice
    mwSize         dim0, dim1;    // MATLAB dims (rows, cols)
    std::string    path;          // destination file
    bool           isXYZ;         // storage order flag
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
};

/* Each thread owns one scratch buffer large enough for the biggest slice. */
static thread_local std::vector<uint8_t> scratch;

/* ───────────────────────────────── SLICE WRITER ─────────────────────────── */
static void save_slice(const SaveTask& t)
{
    const mwSize srcRows = t.isXYZ ? t.dim1 : t.dim0;
    const mwSize srcCols = t.isXYZ ? t.dim0 : t.dim1;
    const uint8_t* inputSlice = t.base + t.offset_bytes;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.isXYZ);

    const uint8_t* ioBuf = nullptr;         // pointer actually passed to libtiff

    /* (1) Prepare IO buffer ­– transpose if slice is [Y X] order or if compressed */
    if (directWrite) {
        ioBuf = inputSlice;                 // happy fast-path
    } else {
        if (scratch.size() < t.bytesPerSlice)
            scratch.resize(t.bytesPerSlice);
        uint8_t* dst = scratch.data();

        if (!t.isXYZ) {                     // transpose [Y X] → [X Y]
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = inputSlice + col * t.dim0 * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (size_t(row) * srcCols + col) * t.bytesPerPixel;
                    std::memcpy(dst + dstIdx,
                                srcColumn + row * t.bytesPerPixel,
                                t.bytesPerPixel);          // FIX: honour 1- or 2-byte pixel
                }
            }
        } else {                            // already [X Y] – just copy row-wise
            const size_t rowBytes = srcCols * t.bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row)
                std::memcpy(dst + row * rowBytes,
                            inputSlice + row * rowBytes,
                            rowBytes);
        }
        ioBuf = dst;
    }

    /* (2) Open TIFF */
    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    /* (3) Baseline tags */
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,     t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

    /* (4) Choose strip size – don’t allocate >2 GB in libtiff when compressed */
    if (t.compressionTag == COMPRESSION_NONE)
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);
    else
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,
                     TIFFDefaultStripSize(tif, 0));   // libtiff default (~8 k)

    /* (5) Write -------------------------------------------------------------- */
    uint8_t* writeBuf = const_cast<uint8_t*>(ioBuf);          // mutable view

    tsize_t wrote = (t.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip    (tif, 0, writeBuf,
                                static_cast<tsize_t>(t.bytesPerSlice))
        : TIFFWriteEncodedStrip(tif, 0, writeBuf,
                                static_cast<tsize_t>(t.bytesPerSlice));

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

/* ──────────────────────────────── THREAD POOL ───────────────────────────── */
namespace {

std::shared_ptr<const std::vector<SaveTask>> g_tasks;
std::atomic_size_t g_next{0};
std::atomic_size_t g_remaining{0};
std::vector<std::string> g_errs;

std::mutex              g_mtx;
std::condition_variable g_cv;
std::atomic<bool>       g_stop{false};
std::vector<std::thread> g_workers;

/* Worker routine ---------------------------------------------------------- */
void worker_loop()
{
    for (;;) {
        std::shared_ptr<const std::vector<SaveTask>> myTasks;

        /* Wait for work or shutdown */
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv.wait(lk, [] { return g_tasks || g_stop.load(); });
            if (g_stop) return;
            myTasks = g_tasks;          // keep a reference so tasks stay alive
        }

        /* Pop indices atomically and process */
        size_t idx = g_next.fetch_add(1, std::memory_order_relaxed);
        while (idx < myTasks->size()) {
            try {
                save_slice((*myTasks)[idx]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> errlk(g_mtx);
                g_errs.emplace_back(e.what());
            }

            if (g_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                /* last slice done; wake main thread */
                std::lock_guard<std::mutex> lk(g_mtx);
                g_tasks.reset();        // mark queue empty
                g_cv.notify_all();
            }
            idx = g_next.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

/* Pool initialiser / grower ---------------------------------------------- */
void ensure_pool(size_t nSlices)
{
    size_t hw = std::thread::hardware_concurrency();
    if (hw <= 1) hw = 8;                        // fall-back heuristic

    size_t want = std::min(nSlices, hw);
    if (hw >= 8) want = std::max<size_t>(8, want);
    if (want == 0) want = 1;

    /* Grow pool if needed */
    std::lock_guard<std::mutex> lk(g_mtx);
    while (g_workers.size() < want)
        g_workers.emplace_back(worker_loop);

    /* Register atexit once */
    static bool exitHooked = false;
    if (!exitHooked) {
        mexAtExit(+[] {
            g_stop.store(true, std::memory_order_relaxed);
            g_cv.notify_all();
            for (auto& t : g_workers)
                if (t.joinable()) t.join();
        });
        exitHooked = true;
    }
}

} // unnamed namespace

/* ───────────────────────────────── MEX ENTRY ────────────────────────────── */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        /* ---- volume ---- */
        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Volume must be uint8 or uint16");

        const mwSize* dims = mxGetDimensions(V);
        const size_t dim0 = dims[0];
        const size_t dim1 = dims[1];
        const size_t dim2 =
            (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId = mxGetClassID(V);
        const size_t bytesPerPixel = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSlice = dim0 * dim1 * bytesPerPixel;

        /* ---- order flag ---- */
        bool isXYZ = false;
        const mxArray* ord = prhs[2];
        if (mxIsLogicalScalar(ord))
            isXYZ = mxIsLogicalScalarTrue(ord);
        else if (mxIsDouble(ord) || mxIsSingle(ord))
            isXYZ = (mxGetScalar(ord) != 0.0);
        else if (mxIsUint32(ord) || mxIsInt32(ord))
            isXYZ = (*static_cast<uint32_t*>(mxGetData(ord)) != 0);

        /* ---- compression ---- */
        char* comp_cstr = mxArrayToUTF8String(prhs[3]);
        if (!comp_cstr)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Failed to parse compression");
        std::string comp(comp_cstr);
        mxFree(comp_cstr);

        uint16_t compTag = COMPRESSION_NONE;
        if (comp == "lzw")      compTag = COMPRESSION_LZW;
        else if (comp == "deflate" || comp == "zip")
                               compTag = COMPRESSION_DEFLATE;
        else if (comp != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Compression must be 'none', 'lzw', or 'deflate'");

        /* ---- file list ---- */
        if (!mxIsCell(prhs[1]) ||
            mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "fileList must be a cell array matching Z-size");

        mxArray** file_cells =
            static_cast<mxArray**>(mxGetData(prhs[1]));

        std::vector<std::string> paths(dim2);
        for (size_t i = 0; i < dim2; ++i) {
            if (!mxIsChar(file_cells[i]))
                mexErrMsgIdAndTxt("save_bl_tif:Input",
                                  "fileList element is not a string");
            char* s = mxArrayToUTF8String(file_cells[i]);
            paths[i] = s; mxFree(s);
        }

        /* ---- build task list ---- */
        auto tasks = std::make_shared<std::vector<SaveTask>>();
        tasks->reserve(dim2);
        for (size_t z = 0; z < dim2; ++z)
            tasks->push_back({ base,
                               z * bytesPerSlice,
                               dim0, dim1,
                               paths[z],
                               isXYZ, classId,
                               compTag,
                               bytesPerSlice,
                               bytesPerPixel });

        if (tasks->empty()) return;

        /* ---- dispatch ---- */
        ensure_pool(tasks->size());

        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_tasks     = tasks;
            g_next      = 0;
            g_remaining = tasks->size();
            g_errs.clear();
        }
        g_cv.notify_all();

        /* Wait for completion */
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv.wait(lk, [] { return g_tasks == nullptr; });
        }

        /* Report errors if any */
        if (!g_errs.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : g_errs)
                msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        /* Optional pass-through return */
        if (nlhs > 0)
            plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
