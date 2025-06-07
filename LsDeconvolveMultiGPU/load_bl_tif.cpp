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
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <exception>

// --- Config ---
constexpr uint16_t kSupportedBitDepth8  = 8;
constexpr uint16_t kSupportedBitDepth16 = 16;

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

// Task description for the queue
struct LoadTask {
    int in_row0, in_col0;   // where to start reading from TIFF (Y, X)
    int out_row0, out_col0; // where to write to output (Y, X)
    int cropH, cropW;       // how many rows/cols to copy
    int roiH, roiW;         // full output block size
    int zIndex;             // output slice index
    void* dstBase;          // output pointer
    int pixelsPerSlice;
    std::string path;       // TIFF file path
    bool transpose;
    LoadTask() = default;
    LoadTask(
        int inY, int inX, int outY, int outX, int h, int w, int roiH_, int roiW_,
        int z, void* dst, int pps, std::string filename, bool transpose_
    ) : in_row0(inY), in_col0(inX), out_row0(outY), out_col0(outX),
        cropH(h), cropW(w), roiH(roiH_), roiW(roiW_),
        zIndex(z), dstBase(dst), pixelsPerSlice(pps), path(std::move(filename)), transpose(transpose_) {}
};

struct TiffCloser {
    void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

inline void getImageSize(TIFF* tif, uint32_t& w, uint32_t& h) {
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
}

inline int computeDstIndex(const LoadTask& task, int row, int col) noexcept {
    if (!task.transpose) {
        return (task.out_row0 + row)
             + (task.out_col0 + col) * task.roiH
             + task.zIndex * task.pixelsPerSlice;
    } else {
        return (task.out_col0 + col)
             + (task.out_row0 + row) * task.roiW
             + task.zIndex * task.pixelsPerSlice;
    }
}

static void swap_uint16_buf(void* buf, int count) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (int i = 0; i < count; ++i) {
        uint16_t v = p[i];
        p[i] = (v >> 8) | (v << 8);
    }
}

// Read/copy a cropped subregion from a TIFF slice
static void copySubRegion(const LoadTask& task, TIFF* tif, uint8_t bytesPerPixel)
{
    uint32_t imgWidth, imgHeight;
    getImageSize(tif, imgWidth, imgHeight);
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    uint32_t tileWidth = 0, tileHeight = 0;
    int isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    }
    if (samplesPerPixel != 1 || (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
        throw std::runtime_error("Only 8/16-bit grayscale TIFFs are supported");

    if (isTiled) {
        tsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0)
            throw std::runtime_error("Invalid tile size returned");
        if (tileWidth == 0 || tileHeight == 0)
            throw std::runtime_error("Invalid tile dimensions in TIFF metadata");
        std::vector<uint8_t> tilebuf(tileSize);
        for (int row = 0; row < task.cropH; ++row) {
            uint32_t imgY = static_cast<uint32_t>(task.in_row0 + row);
            for (int col = 0; col < task.cropW; ++col) {
                uint32_t imgX = static_cast<uint32_t>(task.in_col0 + col);
                uint32_t tileX = (imgX / tileWidth) * tileWidth;
                uint32_t tileY = (imgY / tileHeight) * tileHeight;
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (TIFFReadEncodedTile(tif, tileIdx, tilebuf.data(), tilebuf.size()) < 0)
                    throw std::runtime_error("Failed reading tile");

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                    int n_tile_pixels = tileSize / bytesPerPixel;
                    swap_uint16_buf(tilebuf.data(), n_tile_pixels);
                }
                uint32_t relY = imgY - tileY;
                uint32_t relX = imgX - tileX;
                int rowStride = tileWidth * bytesPerPixel; // Bytes per row in tile
                int offset = relY * rowStride + relX * bytesPerPixel;
                int dstOffset = computeDstIndex(task, row, col) * bytesPerPixel;
                std::memcpy(
                    static_cast<uint8_t*>(task.dstBase) + dstOffset,
                    tilebuf.data() + offset,
                    bytesPerPixel
                );
            }
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;
        int stripBufSize = static_cast<int>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (stripBufSize <= 0 || stripBufSize > (1 << 30)) {
            throw std::runtime_error("Invalid or too large strip buffer size");
        }
        std::vector<uint8_t> stripbuf(stripBufSize);
        tstrip_t currentStrip = (tstrip_t)-1;
        for (int row = 0; row < task.cropH; ++row) {
            uint32_t tifRow = static_cast<uint32_t>(task.in_row0 + row);
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);

            if (stripIdx != currentStrip) {
                tsize_t nbytes = TIFFReadEncodedStrip(tif, stripIdx, stripbuf.data(), stripBufSize);
                if (nbytes < 0)
                    throw std::runtime_error("TIFFReadEncodedStrip failed");
                currentStrip = stripIdx;
            }

            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow = tifRow - stripStartRow;
            uint8_t* scanlinePtr = stripbuf.data() + (relRow * imgWidth * bytesPerPixel);

            if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                swap_uint16_buf(scanlinePtr, imgWidth);
            }

            for (int col = 0; col < task.cropW; ++col) {
                int srcOffset = (task.in_col0 + col) * bytesPerPixel;
                int dstOffset = computeDstIndex(task, row, col) * bytesPerPixel;
                std::memcpy(
                    static_cast<uint8_t*>(task.dstBase) + dstOffset,
                    scanlinePtr + srcOffset,
                    bytesPerPixel
                );
            }
        }
    }
}

// Thread-safe work queue
class TaskQueue {
    std::queue<LoadTask> q;
    std::mutex m;
    std::condition_variable cv;
    bool finished = false;
public:
    void push(LoadTask&& t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(std::move(t));
        cv.notify_one();
    }
    bool pop(LoadTask& t) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return !q.empty() || finished; });
        if (!q.empty()) {
            t = std::move(q.front());
            q.pop();
            return true;
        }
        return false;
    }
    void set_finished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        cv.notify_all();
    }
};

// Worker thread function
void worker_main(TaskQueue* tq, std::vector<std::string>& errors, std::mutex& err_mutex,
                 uint8_t bytesPerPixel) {
    LoadTask task;
    while (tq->pop(task)) {
        try {
            TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
            if (!tif) {
                std::lock_guard<std::mutex> lck(err_mutex);
                errors.emplace_back("Cannot open file " + task.path);
                continue;
            }
            copySubRegion(task, tif.get(), bytesPerPixel);
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back(task.path + ": " + ex.what());
        }
    }
}

// Entry point
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First argument must be a cell array of filenames");

    bool transpose = false;
    if (nrhs == 6) {
        if (!mxIsLogicalScalar(prhs[5]))
            mexErrMsgIdAndTxt("load_bl_tif:Transpose", "transposeFlag must be a logical scalar.");
        transpose = mxIsLogicalScalarTrue(prhs[5]);
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
            mexErrMsgIdAndTxt("load_bl_tif:Input", "Filename in cell %d is empty", i);
        fileList[i] = mstr.get();
    }

    int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    int roiW  = static_cast<int>(mxGetScalar(prhs[4]));

    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI parameters invalid");

    TIFFSetWarningHandler(nullptr);
    TiffHandle tif0(TIFFOpen(fileList[0].c_str(), "r"));
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open file %s (slice 0)", fileList[0].c_str());
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    getImageSize(tif0.get(), imgWidth, imgHeight);
    TIFFGetField(tif0.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif0.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
                          "Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported.");

    const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;

    mwSize outH = transpose ? roiW : roiH;
    mwSize outW = transpose ? roiH : roiW;
    mwSize dims[3] = { outH, outW, static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);

    void* outData = mxGetData(plhs[0]);
    int pixelsPerSlice = static_cast<int>(outH) * outW;
    std::fill_n(static_cast<uint8_t*>(outData), pixelsPerSlice * numSlices * bytesPerPixel, 0);

    // --- Prepare task queue ---
    TaskQueue queue;
    std::vector<std::string> errors;
    std::mutex err_mutex;

    // --- Generate all tasks ---
    for (int z = 0; z < numSlices; ++z)
    {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Cannot open file " + fileList[z]);
            continue;
        }
        getImageSize(tif.get(), imgWidth, imgHeight);

        int img_y_start = std::max(roiY0, 0);
        int img_y_end   = std::min(roiY0 + roiH - 1, static_cast<int>(imgHeight) - 1);
        int img_x_start = std::max(roiX0, 0);
        int img_x_end   = std::min(roiX0 + roiW - 1, static_cast<int>(imgWidth) - 1);

        int cropHz = img_y_end - img_y_start + 1; // rows to copy
        int cropWz = img_x_end - img_x_start + 1; // cols to copy

        if (cropHz <= 0 || cropWz <= 0) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Slice " + std::to_string(z) + " has no overlap with ROI");
            continue;
        }
        int out_row0 = std::max(0, img_y_start - roiY0);
        int out_col0 = std::max(0, img_x_start - roiX0);
        if (out_row0 + cropHz > roiH || out_col0 + cropWz > roiW) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Crop region exceeds output bounds for slice " + std::to_string(z));
            continue;
        }
        queue.push(LoadTask{
            img_y_start, img_x_start,
            out_row0, out_col0,
            cropHz, cropWz,
            roiH, roiW,
            z,
            outData,
            pixelsPerSlice,
            fileList[z],
            transpose
        });
    }

    // --- Launch thread pool ---
#ifdef _WIN32
    // Start with only 1 thread for safety, scale up after validating
    unsigned numThreads = 1;
    const char* env_threads = getenv("LOAD_BL_TIF_THREADS");
    if (env_threads) numThreads = std::max(1u, (unsigned)atoi(env_threads));
#else
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
#endif
    // NUM THREADS (see above) is set in a way safe for Windows
    std::vector<std::thread> workers;
    std::atomic<bool> hasError{false};

    auto do_tasks = [&](int tid) {
        LoadTask task;
        while (!hasError && queue.pop(task)) {
            try {
                TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
                if (!tif) {
                    std::lock_guard<std::mutex> lck(err_mutex);
                    errors.emplace_back("Cannot open file " + task.path);
                    hasError = true; break;
                }
                copySubRegion(task, tif.get(), bytesPerPixel);
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lck(err_mutex);
                errors.emplace_back(task.path + ": " + ex.what());
                hasError = true; break;
            } catch (...) {
                std::lock_guard<std::mutex> lck(err_mutex);
                errors.emplace_back(task.path + ": unknown error");
                hasError = true; break;
            }
        }
    };

    if (numThreads == 1) {
        do_tasks(0);
    } else {
        for (unsigned i = 0; i < numThreads; ++i)
            workers.emplace_back(do_tasks, i);
        queue.set_finished();
        for (auto& t : workers) t.join();
    }

    // --- Error check ---
    if (!errors.empty()) {
        std::string allerr = "Errors during load_bl_tif:\n";
        for (const auto& s : errors) allerr += s + "\n";
        mexErrMsgIdAndTxt("load_bl_tif:Threaded", "%s", allerr.c_str());
    }
}
