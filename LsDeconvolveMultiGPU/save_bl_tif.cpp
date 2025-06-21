/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  VERSION  : 2025-06-21  (no-persistence, per-call pool, eager-bind, pre-scratch)
  AUTHOR   : Keivan Moradi  (with ChatGPT-4o assistance)
  LICENSE  : GNU GPL v3   <https://www.gnu.org/licenses/>
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
#  include <fcntl.h>
#  include <unistd.h>
#endif

/* ───────────────────────────── TASK DESCRIPTION ─────────────────────────── */
struct SaveTask {
    const uint8_t* basePtr;       // start of whole volume
    size_t         sliceOffset;   // byte-offset of this slice
    mwSize         rows, cols;    // MATLAB dims *after* any transpose
    std::string    filePath;      // destination path
    bool           alreadyXYZ;    // true if input is [X Y Z]
    mxClassID      classId;
    uint16_t       compressionTag;
    size_t         bytesPerSlice;
    size_t         bytesPerPixel;
};

/* Each thread owns one reusable scratch buffer large enough for one slice.   */
static thread_local std::vector<uint8_t> scratch;

/* ───────────────────────────── LOW-LEVEL TIFF WRITE ─────────────────────── */
static void save_slice(const SaveTask& t)
{
    const mwSize srcRows = t.alreadyXYZ ? t.cols : t.rows;    // if transpose needed
    const mwSize srcCols = t.alreadyXYZ ? t.rows : t.cols;

    const uint8_t* src = t.basePtr + t.sliceOffset;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.alreadyXYZ);

    const uint8_t* ioBuf = nullptr;          // buffer actually passed to libtiff

    /* 1. Prepare buffer (handle transpose / compression fast-path) */
    if (directWrite) {
        ioBuf = src;                         // zero-copy path 🚀
    } else {
        if (scratch.size() < t.bytesPerSlice)
            scratch.resize(t.bytesPerSlice); // one-time per thread

        uint8_t* dst = scratch.data();

        if (!t.alreadyXYZ) {                 // transpose [Y X] → [X Y]
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = src + col * t.rows * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * t.bytesPerPixel;
                    std::memcpy(dst + dstIdx,
                                srcColumn + row * t.bytesPerPixel,
                                t.bytesPerPixel);
                }
            }
        } else {                            // already XYZ – need copy only for compression
            const size_t rowBytes = srcCols * t.bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row)
                std::memcpy(dst + row * rowBytes,
                            src + row * rowBytes,
                            rowBytes);
        }
        ioBuf = dst;
    }

    /* 2. Open TIFF (write-mode) */
    TIFF* tif = TIFFOpen(t.filePath.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.filePath);

    /* 3. Required baseline tags */
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,     t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

    /* 4. One-strip-per-image (fast and simple) */
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    /* 5. Write the single strip */
    uint8_t* writeBuf = const_cast<uint8_t*>(ioBuf);
    tsize_t  nWritten = (t.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip    (tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice))
        : TIFFWriteEncodedStrip(tif, 0, writeBuf, static_cast<tsize_t>(t.bytesPerSlice));

    if (nWritten < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF write failed on " + t.filePath);
    }

#if defined(__linux__)
    int fd = TIFFFileno(tif);
    if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // drop page cache
#endif
    TIFFClose(tif);
}

/* ───────────────────────────── ONE-SHOT POOL (PER CALL) ─────────────────── */
namespace {

/* Context shared by all threads of *this* mexFunction call */
struct CallContext {
    std::shared_ptr<const std::vector<SaveTask>> tasks;
    std::atomic_size_t nextIndex{0};
    std::mutex errMutex;
    std::vector<std::string> errors;
    size_t maxSliceBytes{0};                 // for scratch pre-allocation
};

/* Worker entry ------------------------------------------------------------ */
void worker_entry(CallContext& ctx)
{
    /* Pre-allocate this thread’s scratch once (largest slice) */
    if (scratch.size() < ctx.maxSliceBytes) scratch.resize(ctx.maxSliceBytes);

    const auto& jobList = *ctx.tasks;
    size_t idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);

    while (idx < jobList.size()) {
        try {
            save_slice(jobList[idx]);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(ctx.errMutex);
            ctx.errors.emplace_back(e.what());
        }
        idx = ctx.nextIndex.fetch_add(1, std::memory_order_relaxed);
    }
}

} // unnamed namespace

/* ──────────────────────────────── MEX ENTRY ─────────────────────────────── */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    try {
        /* ─── 1. ARGUMENT VALIDATION ─────────────────────────────────────── */
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Usage: save_bl_tif(volume, fileList, orderFlag, compression)");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Volume must be uint8 or uint16.");

        const mwSize* dims   = mxGetDimensions(V);
        const size_t  dim0   = dims[0];
        const size_t  dim1   = dims[1];
        const size_t  dim2   = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* basePtr  = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId = mxGetClassID(V);
        const size_t bytesPerPx = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSl = dim0 * dim1 * bytesPerPx;

        /* orderFlag: true  => already [X Y Z];  false => MATLAB [Y X Z] default */
        bool alreadyXYZ = false;
        const mxArray* ord = prhs[2];
        if (mxIsLogicalScalar(ord)) alreadyXYZ = mxIsLogicalScalarTrue(ord);
        else if (mxIsNumeric(ord))  alreadyXYZ = (mxGetScalar(ord) != 0.0);

        /* compression: 'none' | 'lzw' | 'deflate' */
        std::string compStr;
        {
            char* cstr = mxArrayToUTF8String(prhs[3]);
            if (!cstr) mexErrMsgIdAndTxt("save_bl_tif:Input", "Invalid compression string.");
            compStr = cstr; mxFree(cstr);
        }
        uint16_t compTag = COMPRESSION_NONE;
        if (compStr == "lzw")           compTag = COMPRESSION_LZW;
        else if (compStr == "deflate" || compStr == "zip")
                                         compTag = COMPRESSION_DEFLATE;
        else if (compStr != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Compression must be 'none', 'lzw', or 'deflate'.");

        /* fileList cell array (length == dim2) */
        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "fileList must be a cell array matching Z dimension.");

        mxArray** cellPtr = static_cast<mxArray**>(mxGetData(prhs[1]));
        std::vector<std::string> paths(dim2);
        for (size_t z = 0; z < dim2; ++z) {
            if (!mxIsChar(cellPtr[z]))
                mexErrMsgIdAndTxt("save_bl_tif:Input",
                                  "fileList element is not a string.");
            char* s = mxArrayToUTF8String(cellPtr[z]);
            paths[z] = s; mxFree(s);
        }

        if (paths.empty()) return;      // Nothing to do

        /* ─── 2. BUILD TASK VECTOR ───────────────────────────────────────── */
        auto taskVec = std::make_shared<std::vector<SaveTask>>();
        taskVec->reserve(dim2);
        for (size_t z = 0; z < dim2; ++z)
            taskVec->push_back({ basePtr,
                                 z * bytesPerSl,
                                 dim0, dim1,
                                 paths[z],
                                 alreadyXYZ,
                                 classId,
                                 compTag,
                                 bytesPerSl,
                                 bytesPerPx });

        /* ─── 3. EAGER-BIND libtiff symbols (one quick call) ─────────────── */
        {
            TIFF* tmp = TIFFOpen("/dev/null", "r");
            if (tmp) TIFFClose(tmp);
        }

        /* ─── 4. LAUNCH ONE-SHOT THREAD POOL ─────────────────────────────── */
        CallContext ctx;
        ctx.tasks          = std::move(taskVec);
        ctx.maxSliceBytes  = bytesPerSl;

        /* Thread count: do *not* artificially cap per user's request          */
        size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = ctx.tasks->size();            // fallback
        const size_t nThreads = std::min(hw, ctx.tasks->size());

        std::vector<std::thread> workers;
        workers.reserve(nThreads);
        for (size_t i = 0; i < nThreads; ++i)
            workers.emplace_back(worker_entry, std::ref(ctx));

        for (auto& t : workers) t.join();               // wait & release pool

        /* ─── 5. PROPAGATE ERRORS (if any) ───────────────────────────────── */
        if (!ctx.errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : ctx.errors) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }

        /* Optional pass-through return */
        if (nlhs) plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
