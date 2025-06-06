#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>  // for memcpy

typedef unsigned char uint8_T;
typedef unsigned short uint16_T;

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
    if (pixelSize != 1 && pixelSize != 2)
        mexErrMsgIdAndTxt("TIFFLoad:InvalidPixelSize", "Unsupported pixel size: %zu", pixelSize);

    size_t scanlineSize = imgWidth * pixelSize;
    std::vector<uint8_t> rowBuffer(scanlineSize);

    for (int row = 0; row < task.height; ++row) {
        int scanlineIdx = task.y - 1 + row;
        if (!TIFFReadScanline(tif, rowBuffer.data(), scanlineIdx))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline %d in: %s", scanlineIdx, task.filename.c_str());

        for (int col = 0; col < task.width; ++col) {
            size_t srcIdx = static_cast<size_t>(task.x - 1 + col) * pixelSize;

            // Transpose row/col: write as [row, col] → [y, x] → [x, y] in memory
            size_t dstPixelOffset = static_cast<size_t>(row) +            // Y
                                    static_cast<size_t>(col) * task.height + // X
                                    task.zindex * task.planeStride;
            size_t dstByteOffset = dstPixelOffset * pixelSize;

            // DEBUG PRINT: Only on very first pixel for each plane
            if (row == 0 && col == 0) {
                mexPrintf("[DEBUG] %s: scanline=%d, rowBuffer[0..4]={", task.filename.c_str(), scanlineIdx);
                for (int k = 0; k < 5 * pixelSize; ++k)
                    mexPrintf("%s%d", (k ? "," : ""), rowBuffer[k]);
                mexPrintf("} srcIdx=%zu dstPixelOffset=%zu dstByteOffset=%zu\n", srcIdx, dstPixelOffset, dstByteOffset);
                mexEvalString("drawnow;");
            }

            if (task.type == mxUINT8_CLASS || task.type == mxUINT16_CLASS) {
                std::memcpy(static_cast<uint8_t*>(task.dst) + dstByteOffset,
                            rowBuffer.data() + srcIdx,
                            pixelSize);

                // DEBUG: print value just written
                if (row == 0 && col == 0) {
                    if (task.type == mxUINT8_CLASS)
                        mexPrintf("[DEBUG] Written UINT8 value: %u\n", *(uint8_T*)(static_cast<uint8_t*>(task.dst) + dstByteOffset));
                    else
                        mexPrintf("[DEBUG] Written UINT16 value: %u\n", *(uint16_T*)(static_cast<uint8_t*>(task.dst) + dstByteOffset));
                    mexEvalString("drawnow;");
                }
            } else {
                mexErrMsgIdAndTxt("TIFFLoad:UnsupportedType", "Unsupported output data type.");
            }
        }
    }
    TIFFClose(tif);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5)
        mexErrMsgIdAndTxt("TIFFLoad:Usage", "Usage: img = load_bl_tif(files, y, x, height, width [, num_threads])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("TIFFLoad:InvalidInput", "First argument must be a cell array of filenames.");

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

    int y = static_cast<int>(mxGetScalar(prhs[1]));
    int x = static_cast<int>(mxGetScalar(prhs[2]));
    int height = static_cast<int>(mxGetScalar(prhs[3]));
    int width = static_cast<int>(mxGetScalar(prhs[4]));

    // Validate first image metadata
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
    if ((uint32_t)(x + width - 1) >= imgWidth || (uint32_t)(y + height - 1) >= imgHeight)
        mexErrMsgTxt("Requested subregion is out of bounds.");

    mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    mwSize dims[3] = { static_cast<mwSize>(width), static_cast<mwSize>(height), static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    size_t planeStride = static_cast<size_t>(width) * height;

    for (size_t i = 0; i < filenames.size(); ++i) {
        LoadTask task = {
            filenames[i], y, x, height, width,
            i, outData, planeStride, outType
        };
        load_subregion(task);
    }
}
