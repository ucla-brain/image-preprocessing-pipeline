/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads]);

  INPUT:
    volume      : 3D MATLAB array of type uint8 or uint16.
    fileList    : 1×Z cell array of output filenames.
    isXYZ       : scalar logical or numeric. True if array is [X Y Z].
    compression : "none", "lzw", or "deflate".
    nThreads    : (optional) Number of threads (default = hardware concurrency).

  FEATURES:
    • Persistent thread pool for reuse across calls.
    • Thread-safe task queue with atomic progress counter.
    • Scope-based TIFF handle cleanup (closed before rename).
    • Robust filesystem ops via std::filesystem.
    • Per-thread scratch buffers to minimize reallocations.
    • Exception aggregation and reporting.

  LIMITATIONS:
    • Grayscale only (single-channel).
    • Single-strip TIFF per slice.
    • No retry on I/O errors.

  DEPENDENCIES:
    libtiff, MATLAB MEX API, C++17 <filesystem>.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)

  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <cstdint>
#include <cstring>

namespace fs = std::filesystem;

// RAII wrapper to ensure TIFF* is closed promptly
struct TiffHandle {
    TIFF* tif;
    TiffHandle(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffHandle() {
        if (tif) TIFFClose(tif);
    }
};

// ThreadPool with work-queue and progress tracking
class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stopFlag(false), tasksRemaining(0) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        cvTask.wait(lock, [this]{ return stopFlag || !tasks.empty(); });
                        if (stopFlag && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    try { task(); }
                    catch (...) {
                        std::lock_guard<std::mutex> lk(errorMutex);
                        errors.emplace_back(std::current_exception());
                    }
                    if (--tasksRemaining == 0) {
                        std::lock_guard<std::mutex> lk(doneMutex);
                        cvDone.notify_one();
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(queueMutex);
            stopFlag = true;
        }
        cvTask.notify_all();
        for (auto& t : workers) if (t.joinable()) t.join();
    }

    template<typename F>
    void enqueue(F&& f) {
        ++tasksRemaining;
        {
            std::lock_guard<std::mutex> lk(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        cvTask.notify_one();
    }

    // Block until all enqueued tasks have finished
    void waitAll() {
        std::unique_lock<std::mutex> lk(doneMutex);
        cvDone.wait(lk, [this]{ return tasksRemaining.load() == 0; });
    }

    // Rethrow first stored exception, if any
    void rethrowFirstError() {
        std::lock_guard<std::mutex> lk(errorMutex);
        if (!errors.empty()) std::rethrow_exception(errors.front());
    }

private:
    std::vector<std::thread>          workers;
    std::queue<std::function<void()>> tasks;
    std::mutex                        queueMutex, doneMutex, errorMutex;
    std::condition_variable           cvTask, cvDone;
    std::atomic<bool>                 stopFlag;
    std::atomic<size_t>               tasksRemaining;
    std::vector<std::exception_ptr>   errors;
};

// Write one slice to disk: correctly close before rename
static void writeSlice(
    const uint8_t*           volumeData,
    size_t                   sliceIndex,
    size_t                   dim0,
    size_t                   dim1,
    size_t                   bytesPerSample,
    bool                     isXYZ,
    uint16_t                 compression,
    const std::string&       outputPath,
    std::vector<uint8_t>&    scratchBuffer
) {
    const size_t width      = isXYZ ? dim0 : dim1;
    const size_t height     = isXYZ ? dim1 : dim0;
    const size_t sliceBytes = dim0 * dim1 * bytesPerSample;
    const uint8_t* srcBase  = volumeData + sliceIndex * sliceBytes;

    // build temporary filename
    fs::path tmpPath = fs::path(outputPath).concat(".tmp");

    // scope the TIFF handle so it is closed before rename
    {
        TiffHandle tiff(tmpPath.string(), "w");
        TIFF* tif = tiff.tif;

        // set up all required TIFF tags
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,       static_cast<uint32_t>(width));
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,      static_cast<uint32_t>(height));
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,    static_cast<uint16_t>(bytesPerSample * 8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,  static_cast<uint16_t>(1));
        TIFFSetField(tif, TIFFTAG_COMPRESSION,      compression);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,      PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,     static_cast<uint32_t>(height));

        // reorder from MATLAB column-major into row-major in our scratch buffer
        for (size_t r = 0; r < height; ++r) {
            for (size_t c = 0; c < width; ++c) {
                size_t orow = isXYZ ? c : r;
                size_t ocol = isXYZ ? r : c;
                size_t srcOff = (orow + ocol * dim0) * bytesPerSample;
                size_t dstOff = (r * width + c) * bytesPerSample;
                memcpy(scratchBuffer.data() + dstOff,
                       srcBase + srcOff,
                       bytesPerSample);
            }
        }

        // write either raw or encoded strip
        tsize_t written = (compression == COMPRESSION_NONE)
            ? TIFFWriteRawStrip(tif, 0, scratchBuffer.data(), sliceBytes)
            : TIFFWriteEncodedStrip(tif, 0, scratchBuffer.data(), sliceBytes);
        if (written < 0)
            throw std::runtime_error("TIFF write failed for slice " +
                                     std::to_string(sliceIndex));
    } // <- TIFF closed here

    // atomically replace any existing file
    if (fs::exists(outputPath)) fs::remove(outputPath);
    fs::rename(tmpPath, outputPath);
}

// Entry point for the MEX
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:usage",
            "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");

    // validate volume type
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
        mexErrMsgIdAndTxt("save_bl_tif:type",
            "Volume must be uint8 or uint16.");

    // gather dimensions
    const mwSize* dims = mxGetDimensions(prhs[0]);
    size_t dim0 = dims[0], dim1 = dims[1];
    size_t numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3)
                         ? dims[2] : 1;

    // parse transpose flag
    bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                 (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // parse compression string
    char* compCStr = mxArrayToUTF8String(prhs[3]);
    std::string compressionMode(compCStr);
    mxFree(compCStr);

    uint16_t compression = (compressionMode == "none")    ? COMPRESSION_NONE
                         : (compressionMode == "lzw")     ? COMPRESSION_LZW
                         : (compressionMode == "deflate") ? COMPRESSION_DEFLATE
                         : throw std::runtime_error("Invalid compression: " + compressionMode);

    // fileList must match slice count
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numSlices)
        mexErrMsgIdAndTxt("save_bl_tif:files",
            "fileList must have one entry per slice.");

    std::vector<std::string> outputFiles(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        outputFiles[i] = s;
        mxFree(s);
    }

    // raw pointer to data bytes
    const uint8_t* volumeData = static_cast<const uint8_t*>(mxGetData(prhs[0]));
    size_t bytesPerSample = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2 : 1);

    // determine thread count
    size_t hardwareThreads = std::thread::hardware_concurrency();
    if (hardwareThreads == 0) hardwareThreads = 1;
    size_t requested = (nrhs == 5) ? static_cast<size_t>(mxGetScalar(prhs[4]))
                                   : hardwareThreads;
    size_t numThreads = std::min(requested, numSlices);
    if (numThreads < 1) numThreads = 1;

    // prepare one scratch buffer per thread
    std::vector<std::vector<uint8_t>> scratchBuffers(
        numThreads,
        std::vector<uint8_t>(dim0 * dim1 * bytesPerSample)
    );

    // create or get persistent pool
    static ThreadPool pool(numThreads);

    // enqueue all slice-writing tasks
    for (size_t idx = 0; idx < numSlices; ++idx) {
        pool.enqueue([&, idx]() {
            writeSlice(volumeData, idx,
                       dim0, dim1,
                       bytesPerSample,
                       isXYZ,
                       compression,
                       outputFiles[idx],
                       scratchBuffers[idx % numThreads]);
        });
    }

    // wait for completion and check errors
    pool.waitAll();
    try {
        pool.rethrowFirstError();
    } catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:runtime", ex.what());
    }

    // return the volume back if requested
    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
}
