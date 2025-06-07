#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <atomic>
#include <sstream>

// --- Config ---
constexpr uint16_t kSupportedBitDepth8  = 8;
constexpr uint16_t kSupportedBitDepth16 = 16;
constexpr size_t MAX_TIFF_BLOCK_BYTES = 1ull << 30;
constexpr uint64_t HARD_CAP = 8ull << 30;   // ­8 GiB

// RAII wrapper for mxArrayToUTF8String()
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString", "Failed to convert string from mxArray");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

struct LoadTask {
    int in_row0, in_col0, cropH, cropW;
    int roiH, roiW, zIndex;
    int out_row0, out_col0;
    int pixelsPerSlice;
    std::string path;
    bool transpose;
    LoadTask() = default;
    LoadTask(
        int inY, int inX, int outY, int outX, int h, int w, int roiH_, int roiW_,
        int z, int pps, std::string filename, bool transpose_
    ) : in_row0(inY), in_col0(inX), out_row0(outY), out_col0(outX),
        cropH(h), cropW(w), roiH(roiH_), roiW(roiW_),
        zIndex(z), pixelsPerSlice(pps), path(std::move(filename)), transpose(transpose_) {}
};

struct TiffCloser {
    void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// ------------------------------------------------------------------
//  64-bit-safe destination index utility for output array indexing (portable for both shapes)
// ------------------------------------------------------------------
inline size_t computeDstIndex(const LoadTask& task,
                              int row, int col) noexcept
{
    size_t r = static_cast<size_t>(task.out_row0 + row);
    size_t c = static_cast<size_t>(task.out_col0 + col);
    size_t slice = static_cast<size_t>(task.zIndex);

    // MATLAB arrays are column-major and transpose swaps [Y, X] ↔ [X, Y]
    if (!task.transpose)
        return r + c * static_cast<size_t>(task.roiH)
               + slice * static_cast<size_t>(task.pixelsPerSlice);
    else
        return c + r * static_cast<size_t>(task.roiW)
               + slice * static_cast<size_t>(task.pixelsPerSlice);
}


static void swap_uint16_buf(void* buf, int count) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (int i = 0; i < count; ++i) {
        uint16_t v = p[i];
        p[i] = (v >> 8) | (v << 8);
    }
}

// The result buffer for each block
struct TaskResult {
    size_t block_id; // index into task/result vector
    std::vector<uint8_t> data;
    int cropH, cropW;
    TaskResult(size_t id, size_t datasz, int ch, int cw)
        : block_id(id), data(datasz), cropH(ch), cropW(cw) {}
};

// ---------------------------------------------------------------------------
//  Safe sub-region reader for both tiled and stripped TIFFs
// ---------------------------------------------------------------------------
static void readSubRegionToBuffer(
    const LoadTask& task,
    TIFF* tif,
    uint8_t bytesPerPixel,
    std::vector<uint8_t>& blockBuf)
{
    //-----------------------------------------------------------------------
    //  Common metadata checks
    //-----------------------------------------------------------------------
    uint32_t imgWidth, imgHeight;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);

    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 ||
        (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
    {
        mexErrMsgIdAndTxt("load_bl_tif:Type",
            "Unsupported TIFF format: only 8/16-bit grayscale, 1 sample/pixel");
    }
    //-----------------------------------------------------------------------
    //  Choose path
    //-----------------------------------------------------------------------
    const bool isTiled = TIFFIsTiled(tif);

    if (isTiled)
    {
        //-------------------------------------------------------------------
        //  Tiled path
        //-------------------------------------------------------------------
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0)
            mexErrMsgIdAndTxt("load_bl_tif:Tiled:Meta",
                "Invalid tile size in TIFF metadata");

        const size_t uncompressedTileBytes =
            static_cast<size_t>(tileW) * tileH * bytesPerPixel;
        if (uncompressedTileBytes > MAX_TIFF_BLOCK_BYTES)
            mexErrMsgIdAndTxt("load_bl_tif:Tiled:Size", "Tile buffer exceeds sane limit of 1 GiB (tile %u)");

        std::vector<uint8_t> tilebuf(uncompressedTileBytes);
        const size_t nTilePixels = uncompressedTileBytes / bytesPerPixel;

        uint32_t prevTile = UINT32_MAX;

        for (int row = 0; row < task.cropH; ++row)
        {
            uint32_t imgY = static_cast<uint32_t>(task.in_row0 + row);
            for (int col = 0; col < task.cropW; ++col)
            {
                uint32_t imgX = static_cast<uint32_t>(task.in_col0 + col);
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (tileIdx != prevTile)
                {
                    if (TIFFReadEncodedTile(tif, tileIdx,
                                             tilebuf.data(),
                                             static_cast<tsize_t>(uncompressedTileBytes)) < 0)
                        mexErrMsgIdAndTxt("load_bl_tif:Tiled:ReadFail",
                            "TIFFReadEncodedTile failed (tile %u)", tileIdx);

                    if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif))
                        swap_uint16_buf(tilebuf.data(), static_cast<int>(nTilePixels));

                    prevTile = tileIdx;
                }

                uint32_t relY = imgY % tileH;
                uint32_t relX = imgX % tileW;
                size_t   srcOff = (static_cast<size_t>(relY) * tileW + relX) * bytesPerPixel;
                size_t   dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;

                std::memcpy(blockBuf.data() + dstOff,
                            tilebuf.data() + srcOff,
                            bytesPerPixel);
            }
        }
    }
    else
    {
        // -------------------------------------------------------------------
        //  Stripped path (safe)
        // -------------------------------------------------------------------
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;

        // Allocate once for the theoretical max,
        // but only use actual bytes (`nbytes`) for decoding and bounds checks.
        const size_t maxStripBytes =
            static_cast<size_t>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (maxStripBytes > MAX_TIFF_BLOCK_BYTES)
            mexErrMsgIdAndTxt("load_bl_tif:Strip:Size",
                "Strip buffer (>1 GiB) exceeds sane limits");

        std::vector<uint8_t> stripbuf(maxStripBytes);
        tstrip_t currentStrip = (tstrip_t)-1;
        tsize_t  nbytes       = 0;        // byte count of *current* strip

        for (int row = 0; row < task.cropH; ++row)
        {
            uint32_t tifRow   = static_cast<uint32_t>(task.in_row0 + row);
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);

            // (Re)load strip only when it changes
            if (stripIdx != currentStrip)
            {
                nbytes = TIFFReadEncodedStrip(tif, stripIdx,
                                              stripbuf.data(),
                                              static_cast<tsize_t>(maxStripBytes));
                if (nbytes < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:Strip:ReadFail",
                        "TIFFReadEncodedStrip failed (strip %u)", stripIdx);

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif))
                    swap_uint16_buf(stripbuf.data(),
                                    static_cast<int>(nbytes / 2));

                currentStrip = stripIdx;
            }

            /* How many *fully decoded* rows does this strip hold?            */
            const uint32_t rowsInThisStrip =
                static_cast<uint32_t>(nbytes / (imgWidth * bytesPerPixel));

            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow        = tifRow - stripStartRow;

            if (relRow >= rowsInThisStrip)   // corrupt or truncated strip
                mexErrMsgIdAndTxt("load_bl_tif:Strip:Bounds",
                    "Row %u exceeds decoded strip size (strip %u)", tifRow+1, stripIdx);

            uint8_t* scanlinePtr = stripbuf.data() +
                (static_cast<size_t>(relRow) * imgWidth * bytesPerPixel);

            for (int col = 0; col < task.cropW; ++col)
            {
                size_t srcOff = (static_cast<size_t>(task.in_col0 + col)) * bytesPerPixel;
                size_t dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;

                if (srcOff + bytesPerPixel > static_cast<size_t>(nbytes))
                    mexErrMsgIdAndTxt("load_bl_tif:Strip:Bounds",
                        "Column %u exceeds decoded strip size (strip %u)", col+1, stripIdx);

                std::memcpy(blockBuf.data() + dstOff,
                            scanlinePtr + srcOff,
                            bytesPerPixel);
            }
        }
    }
}

void worker_main(const std::vector<LoadTask>& tasks,
                 uint8_t bytesPerPixel,
                 void* outBase,           // ← mxGetData(plhs[0])
                 std::mutex& err_mutex,
                 std::vector<std::string>& errors,
                 std::atomic<size_t>& err_count,
                 size_t begin, size_t end)
{
    std::vector<uint8_t> scratch;

    for (size_t i = begin; i < end; ++i)
    {
        const auto& task = tasks[i];
        TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
        if (!tif) { /* …record error… */ continue; }

        scratch.resize(static_cast<size_t>(task.cropH) *
                       task.cropW * bytesPerPixel);

        readSubRegionToBuffer(task, tif.get(), bytesPerPixel, scratch);

        // ---------- copy scratch → final MATLAB array -----------
        uint8_t* out = static_cast<uint8_t*>(outBase);
        for (int row = 0; row < task.cropH; ++row)
        {
            for (int col = 0; col < task.cropW; ++col)
            {
                size_t dstElem = computeDstIndex(task, row, col);
                size_t dstByte = dstElem * bytesPerPixel;
                size_t srcByte = (static_cast<size_t>(row)*task.cropW + col) * bytesPerPixel;
                std::memcpy(out + dstByte,
                            scratch.data() + srcByte,
                            bytesPerPixel);
            }
        }
    }
}


// ==============================
//       ENTRY POINT  (PATCHED)
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input", "First argument must be a cell array of filenames");

    /* ------------------------------------------------------------------
     * Parse transpose flag (optional)
     * ------------------------------------------------------------------ */
    bool transpose = false;
    if (nrhs == 6) {
        const mxArray* flag = prhs[5];
        if (mxIsLogicalScalar(flag))
            transpose = mxIsLogicalScalarTrue(flag);
        else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1)
            transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("load_bl_tif:Transpose",
                "transposeFlag must be a logical or int32/uint32 scalar.");
    }

    /* ------------------------------------------------------------------
     * Convert MATLAB cell array → std::vector<std::string>
     * ------------------------------------------------------------------ */
    int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
    for (int i = 0; i < numSlices; ++i) {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell))
            mexErrMsgIdAndTxt("load_bl_tif:Input", "File list must contain only strings.");
        MatlabString mstr(cell);
        if (!mstr.get() || !*mstr.get())
            mexErrMsgIdAndTxt("load_bl_tif:Input", "Filename in cell %d is empty", i+1);
        fileList[i] = mstr.get();
    }

    /* ------------------------------------------------------------------
     * Parse scalar ROI arguments (1-based in MATLAB)
     * ------------------------------------------------------------------ */
    for (int i = 1; i <= 4; ++i)
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1)
            mexErrMsgIdAndTxt("load_bl_tif:InputType",
                "Input argument %d must be a real double scalar.", i+1);

    double y_in = mxGetScalar(prhs[1]);
    double x_in = mxGetScalar(prhs[2]);
    double h_in = mxGetScalar(prhs[3]);
    double w_in = mxGetScalar(prhs[4]);

    if (!mxIsFinite(y_in) || !mxIsFinite(x_in) ||
        !mxIsFinite(h_in) || !mxIsFinite(w_in) ||
        y_in < 1  || x_in < 1  || h_in < 1  || w_in < 1)
        mexErrMsgIdAndTxt("load_bl_tif:Negative",
            "y, x, height, width must be positive finite scalars (1-based).");

    size_t roiY0 = static_cast<size_t>(y_in - 1);
    size_t roiX0 = static_cast<size_t>(x_in - 1);
    size_t roiH  = static_cast<size_t>(h_in);
    size_t roiW  = static_cast<size_t>(w_in);

    /* ------------------------------------------------------------------
     * Per-slice metadata validation (unchanged)
     * ------------------------------------------------------------------ */
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    for (int z = 0; z < numSlices; ++z) {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
                "Cannot open file %s (slice %d)", fileList[z].c_str(), z+1);

        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgWidth);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgHeight);
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

        if (samplesPerPixel != 1 ||
            (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
            mexErrMsgIdAndTxt("load_bl_tif:Type",
                "Only 8/16-bit grayscale TIFFs supported. Slice %d (%s)",
                z+1, fileList[z].c_str());

        if (roiY0 + roiH > imgHeight || roiX0 + roiW > imgWidth)
            mexErrMsgIdAndTxt("load_bl_tif:ROI",
                "Requested ROI [%zu:%zu,%zu:%zu] out of bounds for slice %d (%s)",
                roiY0+1, roiY0+roiH, roiX0+1, roiX0+roiW, z+1, fileList[z].c_str());
    }

    /* ------------------------------------------------------------------
     *  PATCH: estimate total memory & choose strategy
     * ------------------------------------------------------------------ */
    const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;

    auto safeMul = [](uint64_t a, uint64_t b, const char* id) -> uint64_t {
        if (a && b > std::numeric_limits<uint64_t>::max() / a)
            mexErrMsgIdAndTxt(id, "Size overflow");
        return a * b;
    };

    uint64_t roiPixels   = safeMul(roiH, roiW,              "load_bl_tif:Overflow");
    uint64_t memPerSlice = safeMul(roiPixels, bytesPerPixel,"load_bl_tif:Overflow");
    uint64_t memNeeded   = safeMul(memPerSlice, numSlices,  "load_bl_tif:Overflow");

    constexpr uint64_t HARD_CAP = 8ull << 30;   // 8 GiB
    bool useStreaming = memNeeded > HARD_CAP;

    /* ------------------------------------------------------------------
     * Create output mxArray  (MATLAB owns the big block)
     * ------------------------------------------------------------------ */
    mwSize outH = transpose ? roiW : roiH;
    mwSize outW = transpose ? roiH : roiW;
    mwSize dims[3] = { outH, outW, static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims,
        (bytesPerPixel==1 ? mxUINT8_CLASS : mxUINT16_CLASS), mxREAL);
    if (!plhs[0])
        mexErrMsgIdAndTxt("load_bl_tif:Alloc", "Failed to allocate output array.");
    void* outData = mxGetData(plhs[0]);
    int pixelsPerSlice = static_cast<int>(static_cast<size_t>(outH) * outW);

    /* ------------------------------------------------------------------
     * STREAMING MODE  (one slice at a time, single scratch buffer)
     * ------------------------------------------------------------------ */
    if (useStreaming)
    {
        std::vector<uint8_t> scratch;
        scratch.resize(static_cast<size_t>(roiPixels) * bytesPerPixel);

        for (int z = 0; z < numSlices; ++z)
        {
            LoadTask task(
                static_cast<int>(roiY0),            // in_row0
                static_cast<int>(roiX0),            // in_col0
                0, 0,                               // out_row0,out_col0
                static_cast<int>(roiH),
                static_cast<int>(roiW),
                static_cast<int>(roiH),
                static_cast<int>(roiW),
                z,
                pixelsPerSlice,
                fileList[z],
                transpose
            );

            TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
            if (!tif)
                mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
                    "Cannot open file %s (slice %d)", fileList[z].c_str(), z+1);

            readSubRegionToBuffer(task, tif.get(), bytesPerPixel, scratch);

            /* copy into MATLAB array */
            for (size_t row = 0; row < task.cropH; ++row)
                for (size_t col = 0; col < task.cropW; ++col)
                {
                    size_t dstElem = computeDstIndex(task, row, col);
                    size_t dstByte = dstElem * bytesPerPixel;
                    size_t srcByte = (row * task.cropW + col) * bytesPerPixel;
                    std::memcpy(static_cast<uint8_t*>(outData) + dstByte,
                                scratch.data() + srcByte,
                                bytesPerPixel);
                }
        }
        return;    // ---- all done in streaming path ----
    }

    /* ------------------------------------------------------------------
     * THREAD-POOL MODE  (original fast path, mostly untouched)
     * ------------------------------------------------------------------ */
    std::vector<LoadTask> tasks;
    std::vector<TaskResult> results;
    std::vector<std::string> errors;
    std::mutex err_mutex;

    for (int z = 0; z < numSlices; ++z) {
        tasks.emplace_back(
            static_cast<int>(roiY0), static_cast<int>(roiX0),
            0, 0,
            static_cast<int>(roiH), static_cast<int>(roiW),
            static_cast<int>(roiH), static_cast<int>(roiW),
            z,
            pixelsPerSlice,
            fileList[z],
            transpose
        );
        results.emplace_back(results.size(),
            static_cast<size_t>(roiPixels) * bytesPerPixel,
            static_cast<int>(roiH), static_cast<int>(roiW));
    }

    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
    if (const char* env = getenv("LOAD_BL_TIF_THREADS"))
        numThreads = std::max(1u, static_cast<unsigned>(atoi(env)));
#endif
    size_t n_tasks = tasks.size();
    size_t chunk   = (n_tasks + numThreads - 1) / numThreads;
    std::atomic<size_t> error_count{0};
    std::vector<std::thread> workers;

    for (unsigned t = 0; t < numThreads; ++t) {
        size_t begin = t * chunk;
        size_t end   = std::min(n_tasks, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back(worker_main,
            std::cref(tasks),
            bytesPerPixel,
            outData,
            std::ref(err_mutex),
            std::ref(errors),
            std::ref(error_count),
            begin, end);
    }
    for (auto& w : workers) w.join();

    if (error_count > 0) {
        std::ostringstream allerr;
        allerr << "Errors during load_bl_tif:\n";
        for (const auto& s : errors) allerr << "  - " << s << "\n";
        mexErrMsgIdAndTxt("load_bl_tif:Threaded", "%s", allerr.str().c_str());
    }

    /* copy results → mxArray (unchanged) */
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        const auto& res  = results[i];
        for (size_t row = 0; row < task.cropH; ++row)
            for (size_t col = 0; col < task.cropW; ++col)
            {
                size_t dstElem = computeDstIndex(task, row, col);
                size_t dstByte = dstElem * bytesPerPixel;
                size_t srcByte = (row * task.cropW + col) * bytesPerPixel;
                std::memcpy(static_cast<uint8_t*>(outData) + dstByte,
                            res.data.data() + srcByte,
                            bytesPerPixel);
            }
    }
}
