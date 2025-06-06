#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstdint>

typedef unsigned char uint8_T;
typedef unsigned short uint16_T;
using uint8_ptr = uint8_T*;
using uint16_ptr = uint16_T*;

struct LoadTask {
    std::string filename;
    int y, x, height, width;
    size_t zindex;
    void* dst;
    size_t planeStride;
    mxClassID type;
};

void load_subregion(const LoadTask& task) {
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("TIFFLoad:OpenFail", "Failed to open: %s", task.filename.c_str());

    uint32_t imgWidth, imgHeight;
    uint16_t bitsPerSample, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1)
        mexErrMsgIdAndTxt("TIFFLoad:NotGrayscale", "Only grayscale TIFFs are supported: %s", task.filename.c_str());
    if (bitsPerSample != 8 && bitsPerSample != 16)
        mexErrMsgIdAndTxt("TIFFLoad:UnsupportedDepth", "Only 8/16-bit TIFFs are supported.");
    if ((uint32_t)(task.x + task.width) > imgWidth || (uint32_t)(task.y + task.height) > imgHeight)
        mexErrMsgIdAndTxt("TIFFLoad:SubregionBounds", "Subregion out of bounds in: %s", task.filename.c_str());

    size_t pixelSize = bitsPerSample / 8;
    size_t scanlineSize = imgWidth * pixelSize;
    std::vector<uint8_t> rowBuffer(scanlineSize);

    for (int row = 0; row < task.height; ++row) {
        if (!TIFFReadScanline(tif, rowBuffer.data(), task.y + row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline in: %s", task.filename.c_str());

        for (int col = 0; col < task.width; ++col) {
            size_t srcIdx = static_cast<size_t>(task.x + col) * pixelSize;
            size_t dstIdx = static_cast<size_t>(row) + static_cast<size_t>(col) * task.height + task.zindex * task.planeStride;

            if (task.type == mxUINT8_CLASS) {
                ((uint8_ptr)task.dst)[dstIdx] = rowBuffer[srcIdx];
            } else if (task.type == mxUINT16_CLASS) {
                ((uint16_ptr)task.dst)[dstIdx] = ((uint16_T*)rowBuffer.data())[task.x + col];
            }
        }
    }

    TIFFClose(tif);
}

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    if (nrhs == 0)
        mexErrMsgTxt("Usage: img = load_bl_tif(files, y, x, height, width [, num_threads])");
    if (nrhs < 5)
        mexErrMsgTxt("Expected at least 5 input arguments.");

    if (!mxIsCell(prhs[0]))
        mexErrMsgTxt("First argument must be a cell array of filenames.");
    size_t numSlices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> filenames(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        if (!mxIsChar(mxGetCell(prhs[0], i)))
            mexErrMsgIdAndTxt("TIFFLoad:InvalidCell", "File list must contain string elements at index %zu", i);
        char* fname = mxArrayToString(mxGetCell(prhs[0], i));
        if (!fname) mexErrMsgTxt("Invalid filename input.");
        filenames[i] = fname;
        if (filenames[i].empty())
            mexErrMsgIdAndTxt("TIFFLoad:EmptyPath", "Filename at index %zu is empty.", i);
        mxFree(fname);
    }

    int y = (int)mxGetScalar(prhs[1]);
    int x = (int)mxGetScalar(prhs[2]);
    int height = (int)mxGetScalar(prhs[3]);
    int width = (int)mxGetScalar(prhs[4]);

    int concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) concurrency = 4;
    int nthreads = (nrhs >= 6) ? (int)mxGetScalar(prhs[5]) : std::min<int>(concurrency, static_cast<int>(numSlices));
    if (nthreads < 1) nthreads = 1;

    TIFF* tif = TIFFOpen(filenames[0].c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("TIFFLoad:OpenFail", "Failed to open: %s", filenames[0].c_str());

    uint32_t imgWidth, imgHeight;
    uint16_t bitsPerSample, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFClose(tif);

    if (samplesPerPixel != 1)
        mexErrMsgIdAndTxt("TIFFLoad:GrayscaleOnly", "Only grayscale TIFFs are supported.");
    if (bitsPerSample != 8 && bitsPerSample != 16)
        mexErrMsgTxt("Only 8-bit or 16-bit grayscale TIFFs are supported.");
    if ((uint32_t)(x + width) > imgWidth || (uint32_t)(y + height) > imgHeight)
        mexErrMsgTxt("Requested subregion is out of bounds.");

    mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    mwSize dims[3] = { (mwSize)height, (mwSize)width, (mwSize)numSlices };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    const size_t planeStride = static_cast<size_t>(height) * width;

    std::queue<LoadTask> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool done = false;

    for (size_t i = 0; i < filenames.size(); ++i) {
        task_queue.push({
            filenames[i], y, x, height, width,
            i, outData, planeStride, outType
        });
    }

    auto worker = [&]() {
        while (true) {
            LoadTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [&]() { return !task_queue.empty() || done; });
                if (task_queue.empty()) return;
                task = task_queue.front();
                task_queue.pop();
            }
            load_subregion(task);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < nthreads; ++i)
        threads.emplace_back(worker);

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        done = true;
    }
    cv.notify_all();

    for (auto& t : threads)
        t.join();
}
