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
            mexErrMsgIdAndTxt("load_bl_tif:Tiled:Size",
                "Tile buffer (>1 GiB) exceeds sane limits");

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

        /* allocate *once* for the theoretical max, but
           always use nbytes (actual) for swapping / bounds */
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

void worker_main(
    const std::vector<LoadTask>& tasks,
    std::vector<TaskResult>& results,
    uint8_t bytesPerPixel,
    std::mutex& err_mutex,
    std::vector<std::string>& errors,
    std::atomic<size_t>& error_count,
    size_t begin,
    size_t end)
{
    for (size_t i = begin; i < end; ++i) {
        const auto& task = tasks[i];
        try {
            TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
            if (!tif) {
                std::lock_guard<std::mutex> lck(err_mutex);
                errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": Cannot open file " + task.path);
                error_count++;
                continue;
            }
            readSubRegionToBuffer(task, tif.get(), bytesPerPixel, results[i].data);
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": " + ex.what());
            error_count++;
        } catch (...) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": Unknown exception in thread");
            error_count++;
        }
    }
}

// ==============================
//       ENTRY POINT
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input", "First argument must be a cell array of filenames");

    bool transpose = false;
    if (nrhs == 6) {
        const mxArray* flag = prhs[5];

        if (mxIsLogicalScalar(flag)) {
            transpose = mxIsLogicalScalarTrue(flag);
        } else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1) {
            transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        } else {
            mexErrMsgIdAndTxt("load_bl_tif:Transpose",
                "transposeFlag must be a logical or int32/uint32 scalar.");
        }
    }

    int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
    fileList.reserve(numSlices);
    for (int i = 0; i < numSlices; ++i)
    {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell))
            mexErrMsgIdAndTxt("load_bl_tif:Input", "File list must contain only strings.");
        MatlabString mstr(cell);
        if (!mstr.get() || !*mstr.get())
            mexErrMsgIdAndTxt("load_bl_tif:Input", "Filename in cell %d is empty", i+1);
        fileList[i] = mstr.get();
    }
    for (int i = 1; i <= 4; ++i) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1)
            mexErrMsgIdAndTxt("load_bl_tif:InputType",
                "Input argument %d must be a real double scalar.", i+1);
    }

    int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    int roiW  = static_cast<int>(mxGetScalar(prhs[4]));

    // --- PATCH: Robustly validate ROI for all slices BEFORE allocation ---
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    for (int z = 0; z < numSlices; ++z) {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open file %s (slice %d)", fileList[z].c_str(), z+1);

        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgWidth);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgHeight);
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

        if (samplesPerPixel != 1 ||
            (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16)) {
            mexErrMsgIdAndTxt("load_bl_tif:Type",
                "Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported. Slice %d (%s)",
                z+1, fileList[z].c_str());
        }

        // PATCH: Require requested ROI to be fully inside TIFF bounds for all slices
        if (roiY0 < 0 || roiX0 < 0 ||
            roiY0 + roiH > (int)imgHeight ||
            roiX0 + roiW > (int)imgWidth) {
            mexErrMsgIdAndTxt("load_bl_tif:ROI",
                "Requested ROI [%d:%d,%d:%d] is out of bounds for slice %d (file: %s)",
                roiY0+1, roiY0+roiH, roiX0+1, roiX0+roiW, z+1, fileList[z].c_str());
        }
    }

    // --- After validation, all slices are good ---

    const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;

    mwSize outH = transpose ? roiW : roiH;
    mwSize outW = transpose ? roiH : roiW;
    mwSize dims[3] = { outH, outW, static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    if (!plhs[0])
        mexErrMsgIdAndTxt("load_bl_tif:Alloc", "Failed to allocate output array.");

    void* outData = mxGetData(plhs[0]);
    int pixelsPerSlice = static_cast<int>(static_cast<size_t>(outH) * outW);
    std::fill_n(static_cast<uint8_t*>(outData), pixelsPerSlice * numSlices * bytesPerPixel, 0);

    // --- Prepare task list (one per Z) ---
    std::vector<LoadTask> tasks;
    std::vector<TaskResult> results;
    std::vector<std::string> errors;
    std::mutex err_mutex;

    // All slices are valid; populate tasks and results
    for (int z = 0; z < numSlices; ++z)
    {
        // No need to clip, ROI is within TIFF bounds
        int img_y_start = roiY0;
        int img_x_start = roiX0;
        int cropHz = roiH;
        int cropWz = roiW;
        int out_row0 = 0;
        int out_col0 = 0;

        tasks.emplace_back(
            img_y_start, img_x_start,
            out_row0, out_col0,
            cropHz, cropWz,
            roiH, roiW,
            z,
            pixelsPerSlice,
            fileList[z],
            transpose
        );
        results.emplace_back(results.size(), static_cast<size_t>(cropHz * cropWz * bytesPerPixel), cropHz, cropWz);
    }

    // --- Parallel Read ---
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
    const char* env_threads = getenv("LOAD_BL_TIF_THREADS");
    if (env_threads) numThreads = std::max(1u, (unsigned)atoi(env_threads));
#endif

    std::vector<std::thread> workers;
    size_t n_tasks = tasks.size();
    std::atomic<size_t> error_count{0};
    if (n_tasks > 0) {
        size_t chunk = (n_tasks + numThreads - 1) / numThreads;
        for (unsigned t = 0; t < numThreads; ++t) {
            size_t begin = t * chunk;
            size_t end   = std::min(n_tasks, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back(worker_main,
                std::cref(tasks),
                std::ref(results),
                bytesPerPixel,
                std::ref(err_mutex),
                std::ref(errors),
                std::ref(error_count),
                begin, end
            );
        }
        for (auto& w : workers) w.join();
    }

    if (error_count > 0) {
        std::ostringstream allerr;
        allerr << "Errors during load_bl_tif:\n";
        for (const auto& s : errors) {
            allerr << "  - " << s << "\n";  // optional bullet for readability
        }
        mexErrMsgIdAndTxt("load_bl_tif:Threaded", "%s", allerr.str().c_str());
    }

    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        const auto& res  = results[i];
        for (size_t row = 0; row < task.cropH; ++row) {
            for (size_t  col = 0; col < task.cropW; ++col) {
                size_t dstElem = computeDstIndex(task, row, col);
                size_t dstByte = dstElem * bytesPerPixel;
                size_t srcByte = (row * task.cropW + col) * bytesPerPixel;
                std::memcpy(static_cast<uint8_t*>(outData) + dstByte,
                            res.data.data() + srcByte,
                            bytesPerPixel
                );
            }
        }
    }
}
