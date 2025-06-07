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
constexpr size_t kMaxPixelsPerSlice = static_cast<size_t>(std::numeric_limits<int>::max());

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
    size_t in_row0, in_col0, cropH, cropW;
    size_t roiH, roiW, zIndex;
    size_t out_row0, out_col0;
    size_t pixelsPerSlice;
    std::string path;
    bool transpose;
    LoadTask() = default;
    LoadTask(
        size_t inY, size_t inX, size_t outY, size_t outX,
        size_t h, size_t w, size_t roiH_, size_t roiW_,
        size_t z, size_t pps, std::string filename, bool transpose_
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
                              size_t row, size_t col) noexcept
{
    size_t r = task.out_row0 + row;
    size_t c = task.out_col0 + col;
    size_t slice = task.zIndex;

    // MATLAB arrays are column-major and transpose swaps [Y, X] ↔ [X, Y]
    if (!task.transpose)
        return r + c * task.roiH + slice * task.pixelsPerSlice;
    else
        return c + r * task.roiW + slice * task.pixelsPerSlice;
}


static void swap_uint16_buf(void* buf, size_t count) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < count; ++i) {
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
    size_t imgWidth, imgHeight;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight))
    {
        throw std::runtime_error("Missing TIFFTAG_IMAGEWIDTH or IMAGELENGTH in file: " + task.path);
    }

    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 ||
        (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
    {
        throw std::runtime_error("Unsupported TIFF format: only 8/16-bit grayscale, 1 sample/pixel in file: " + task.path);
    }

    const bool isTiled = TIFFIsTiled(tif);

    if (isTiled)
    {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0)
            throw std::runtime_error("Invalid tile size in TIFF metadata in file: " + task.path);

        const size_t uncompressedTileBytes =
            static_cast<size_t>(tileW) * tileH * bytesPerPixel;
        if (uncompressedTileBytes > MAX_TIFF_BLOCK_BYTES)
            throw std::runtime_error("Tile buffer exceeds sane limit of 1 GiB in file: " + task.path);

        std::vector<uint8_t> tilebuf(uncompressedTileBytes);
        const size_t nTilePixels = uncompressedTileBytes / bytesPerPixel;

        uint32_t prevTile = UINT32_MAX;

        for (int row = 0; row < task.cropH; ++row) {
            uint32_t imgY = static_cast<uint32_t>(task.in_row0 + row);
            for (int col = 0; col < task.cropW; ++col) {
                uint32_t imgX = static_cast<uint32_t>(task.in_col0 + col);
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);



                if (tileIdx != prevTile) {
                    tsize_t ret = TIFFReadEncodedTile(
                        tif,
                        tileIdx,
                        tilebuf.data(),
                        static_cast<tsize_t>(uncompressedTileBytes)
                    );
                    if (ret < 0)
                    {
                        std::ostringstream oss;
                        oss << "TIFFReadEncodedTile failed (tile " << tileIdx << ") in file: " << task.path;
                        throw std::runtime_error(oss.str());
                    }
                    size_t validBytes = static_cast<size_t>(ret);
                    size_t validPixels = validBytes / bytesPerPixel;
                    if (bytesPerPixel == 2 && !TIFFIsByteSwapped(tif))
                        swap_uint16_buf(tilebuf.data(), validPixels);

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
        size_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;

        const size_t maxStripBytes =
            static_cast<size_t>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (maxStripBytes > MAX_TIFF_BLOCK_BYTES)
            throw std::runtime_error("Strip buffer exceeds sane limits (>1 GiB) in file: " + task.path);

        std::vector<uint8_t> stripbuf(maxStripBytes);
        tstrip_t currentStrip = (tstrip_t)-1;
        tsize_t  nbytes = 0;

        for (int row = 0; row < task.cropH; ++row)
        {
            uint32_t tifRow   = static_cast<uint32_t>(task.in_row0 + row);
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);

            if (stripIdx != currentStrip) {
                nbytes = TIFFReadEncodedStrip(tif, stripIdx, stripbuf.data(),
                                              static_cast<tsize_t>(maxStripBytes));
                if (nbytes < 0) {
                    std::ostringstream oss;
                    oss << "TIFFReadEncodedStrip failed (strip " << stripIdx << ") in file: " << task.path;
                    throw std::runtime_error(oss.str());
                }

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif))
                    swap_uint16_buf(stripbuf.data(), static_cast<size_t>(nbytes / 2));

                currentStrip = stripIdx;
            }

            const uint32_t rowsInThisStrip =
                static_cast<uint32_t>(nbytes / (imgWidth * bytesPerPixel));
            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow        = tifRow - stripStartRow;

            if (relRow >= rowsInThisStrip) {
                std::ostringstream oss;
                oss << "Row " << tifRow+1 << " exceeds decoded strip size (strip " << stripIdx << ") in file: " << task.path;
                throw std::runtime_error(oss.str());
            }

            uint8_t* scanlinePtr = stripbuf.data() +
                (static_cast<size_t>(relRow) * imgWidth * bytesPerPixel);

            for (int col = 0; col < task.cropW; ++col)
            {
                size_t srcOff = (static_cast<size_t>(task.in_col0 + col)) * bytesPerPixel;
                size_t dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;

                if (srcOff + bytesPerPixel > static_cast<size_t>(nbytes)) {
                    std::ostringstream oss;
                    oss << "Column " << col+1 << " exceeds decoded strip size (strip " << stripIdx << ") in file: " << task.path;
                    throw std::runtime_error(oss.str());
                }

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

    // MATLAB stores [row, col, z], but user may want [col, row, z] (transpose=true)
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

    size_t numSlices = static_cast<size_t>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
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

    double y_in = mxGetScalar(prhs[1]);
    double x_in = mxGetScalar(prhs[2]);
    double h_in = mxGetScalar(prhs[3]);
    double w_in = mxGetScalar(prhs[4]);

    if (!mxIsFinite(y_in) || !mxIsFinite(x_in) ||
        !mxIsFinite(h_in) || !mxIsFinite(w_in))
        mexErrMsgIdAndTxt("load_bl_tif:NaN",
            "y, x, height, width must be finite numbers.");

    if (y_in < 1 || x_in < 1 || h_in < 1 || w_in < 1)
        mexErrMsgIdAndTxt("load_bl_tif:Negative",
            "y, x, height, width must be positive (1-based).");

    auto roiY0 = static_cast<size_t>(y_in - 1);
    auto roiX0 = static_cast<size_t>(x_in - 1);
    auto roiH  = static_cast<size_t>(h_in);
    auto roiW  = static_cast<size_t>(w_in);

    // --- Robustly validate ROI for all slices BEFORE allocation ---
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, globalBitsPerSample = 0, samplesPerPixel = 1;
    for (size_t z = 0; z < numSlices; ++z) {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif) {
            std::ostringstream oss;
            oss << "Cannot open file " << fileList[z] << " (slice " << z+1 << ")";
            mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "%s", oss.str().c_str());
        }
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgWidth);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgHeight);
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

        if (z == 0) {
            globalBitsPerSample = bitsPerSample;
        } else if (bitsPerSample != globalBitsPerSample) {
            mexErrMsgIdAndTxt("load_bl_tif:BitDepthMismatch",
                "Inconsistent bitsPerSample across slices. Expected %u, got %u in slice %d (%s)",
                globalBitsPerSample, bitsPerSample, z+1, fileList[z].c_str());
        }

        if (samplesPerPixel != 1 ||
            (globalBitsPerSample != kSupportedBitDepth8 && globalBitsPerSample != kSupportedBitDepth16)) {
            mexErrMsgIdAndTxt("load_bl_tif:Type",
                "Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported. Slice %d (%s)",
                z+1, fileList[z].c_str());
        }

        // PATCH: Require requested ROI to be fully inside TIFF bounds for all slices
        if (roiY0 + roiH > imgHeight ||
            roiX0 + roiW > imgWidth) {
            mexErrMsgIdAndTxt("load_bl_tif:ROI",
                "Requested ROI [%d:%d,%d:%d] is out of bounds for slice %d (file: %s)",
                roiY0+1, roiY0+roiH, roiX0+1, roiX0+roiW, z+1, fileList[z].c_str());
        }
    }

    // --- After validation, all slices are good ---

    const mxClassID outType = (globalBitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const uint8_t bytesPerPixel = (globalBitsPerSample == 16) ? 2 : 1;

    size_t outH = transpose ? roiW : roiH;
    size_t outW = transpose ? roiH : roiW;
    size_t dims[3] = { outH, outW, numSlices };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    if (!plhs[0])
        mexErrMsgIdAndTxt("load_bl_tif:Alloc", "Failed to allocate output array.");

    void* outData = mxGetData(plhs[0]);
    size_t pixelsPerSlice = outH * outW;
    if (pixelsPerSlice > kMaxPixelsPerSlice)
        mexErrMsgIdAndTxt("load_bl_tif:TooLarge", "Requested ROI too large (>2^31 elements).");

    // --- Prepare task list (one per Z) ---
    std::vector<LoadTask> tasks;
    tasks.reserve(numSlices);
    std::vector<TaskResult> results;
    results.reserve(numSlices);
    std::vector<std::string> errors;
    std::mutex err_mutex;

    // All slices are valid; populate tasks and results
    for (size_t z = 0; z < numSlices; ++z)
    {
        // No need to clip, ROI is within TIFF bounds
        size_t img_y_start = roiY0;
        size_t img_x_start = roiX0;
        size_t cropHz = roiH;
        size_t cropWz = roiW;
        size_t out_row0 = 0;
        size_t out_col0 = 0;

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
    if (env_threads) numThreads = std::max(8u, (unsigned)atoi(env_threads));
#endif

    std::vector<std::thread> workers;
    size_t n_tasks = tasks.size();
    std::atomic<size_t> error_count{0};
    if (n_tasks > 0) {
        size_t chunk = (n_tasks + numThreads - 1) / numThreads;
        for (unsigned t = 0; t < numThreads; ++t) {
            size_t begin = t * chunk;
            size_t end   = std::min(n_tasks, begin + chunk);
            if (begin >= end) break; // No more tasks for this thread
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
            for (size_t col = 0; col < task.cropW; ++col) {
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
