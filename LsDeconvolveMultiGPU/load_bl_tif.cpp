#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <mutex>
#include <cstdint>

typedef unsigned char uint8_T;
typedef unsigned short uint16_T;

struct LoadTask {
    std::string filename;
    int y, x, height, width;
    int zindex;
    void* dst;
    int dst_stride;
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
    if ((bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("TIFFLoad:UnsupportedDepth", "Only 8/16-bit TIFFs are supported.");

    if ((uint32_t)(task.x + task.width) > imgWidth || (uint32_t)(task.y + task.height) > imgHeight)
        mexErrMsgIdAndTxt("TIFFLoad:SubregionBounds", "Subregion out of bounds in: %s", task.filename.c_str());

    size_t pixelSize = bitsPerSample / 8;
    size_t scanlineSize = imgWidth * samplesPerPixel * pixelSize;
    std::vector<uint8_t> rowBuffer(scanlineSize);

    for (int row = 0; row < task.height; ++row) {
        if (!TIFFReadScanline(tif, rowBuffer.data(), task.y + row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline in: %s", task.filename.c_str());

        for (int col = 0; col < task.width; ++col) {
            size_t srcIdx = (task.x + col) * pixelSize;
            size_t dstIdx = row + col * task.height + task.zindex * task.dst_stride;

            if (task.type == mxUINT8_CLASS) {
                if (srcIdx >= rowBuffer.size())
                    mexErrMsgTxt("Read access out of scanline bounds (uint8).");
                ((uint8_T*)task.dst)[dstIdx] = rowBuffer[srcIdx];
            } else {
                if ((task.x + col) >= imgWidth)
                    mexErrMsgTxt("Read access out of scanline bounds (uint16).");
                ((uint16_T*)task.dst)[dstIdx] = ((uint16_T*)rowBuffer.data())[task.x + col];
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
        char* fname = mxArrayToString(mxGetCell(prhs[0], i));
        if (!fname) mexErrMsgTxt("Invalid filename input.");
        filenames[i] = fname;
        mxFree(fname);
    }

    int y = (int)mxGetScalar(prhs[1]);
    int x = (int)mxGetScalar(prhs[2]);
    int height = (int)mxGetScalar(prhs[3]);
    int width = (int)mxGetScalar(prhs[4]);
    int nthreads = (nrhs >= 6) ? (int)mxGetScalar(prhs[5]) : std::min<int>(std::thread::hardware_concurrency(), numSlices);
    if (nthreads < 1) nthreads = 1;

    // Probe first image for type
    TIFF* tif = TIFFOpen(filenames[0].c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("TIFFLoad:OpenFail", "Failed to open: %s", filenames[0].c_str());

    uint32_t imgWidth, imgHeight;
    uint16_t bitsPerSample, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1)
        mexErrMsgTxt("Only grayscale TIFFs are supported.");
    if (bitsPerSample != 8 && bitsPerSample != 16)
        mexErrMsgTxt("Only 8-bit or 16-bit grayscale TIFFs are supported.");
    if ((uint32_t)(x + width) > imgWidth || (uint32_t)(y + height) > imgHeight)
        mexErrMsgTxt("Requested subregion is out of bounds.");

    mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    TIFFClose(tif);

    // Allocate output
    mwSize dims[3] = { (mwSize)height, (mwSize)width, (mwSize)numSlices };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    int stride = height * width;

    // Prepare threaded tasks
    std::vector<std::future<void>> futures;
    std::mutex queue_mutex;
    size_t task_idx = 0;

    auto worker = [&]() {
        while (true) {
            size_t my_idx;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (task_idx >= filenames.size()) return;
                my_idx = task_idx++;
            }
            LoadTask task = { filenames[my_idx], y, x, height, width,
                              (int)my_idx, outData, stride, outType };
            load_subregion(task);
        }
    };

    for (int i = 0; i < nthreads; ++i)
        futures.push_back(std::async(std::launch::async, worker));
    for (auto& f : futures)
        f.get();
}
