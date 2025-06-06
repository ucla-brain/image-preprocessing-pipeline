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

    mexPrintf("\n[DEBUG] ====== LoadTask Start ======\n");
    mexPrintf("[DEBUG]  File      : %s\n", task.filename.c_str());
    mexPrintf("[DEBUG]  x/y/w/h   : x=%d y=%d width=%d height=%d\n", task.x, task.y, task.width, task.height);
    mexPrintf("[DEBUG]  zindex    : %zu\n", task.zindex);
    mexPrintf("[DEBUG]  out type  : %d (8=UINT8, 16=UINT16)\n", task.type);
    mexPrintf("[DEBUG]  imgWidth  : %u\n", imgWidth);
    mexPrintf("[DEBUG]  imgHeight : %u\n", imgHeight);
    mexPrintf("[DEBUG]  bits/sample: %u\n", bitsPerSample);
    mexPrintf("[DEBUG]  pixelSize : %zu\n", bitsPerSample / 8);
    mexPrintf("[DEBUG]  planeStride: %zu\n", task.planeStride);

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
        int tif_row = task.y - 1 + row;
        if (!TIFFReadScanline(tif, rowBuffer.data(), tif_row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline in: %s", task.filename.c_str());

        // Print first 10 values of the row buffer for debugging
        mexPrintf("[DEBUG]  Row %d (tif_row=%d): rowBuffer[0..9]: ", row, tif_row);
        for (size_t i = 0; i < 10 * pixelSize && i < rowBuffer.size(); ++i) {
            mexPrintf("%u ", rowBuffer[i]);
        }
        mexPrintf("\n");

        for (int col = 0; col < task.width; ++col) {
            size_t srcIdx = static_cast<size_t>(task.x + col) * pixelSize;

            size_t dstPixelOffset = static_cast<size_t>(row) +
                                    static_cast<size_t>(col) * task.height +
                                    task.zindex * task.planeStride;
            size_t dstByteOffset = dstPixelOffset * pixelSize;

            // Print per-pixel debug
            if (row < 3 && col < 3) { // Only for a small region to avoid overflood
                mexPrintf("[DEBUG]   row=%d col=%d srcIdx=%zu dstPixelOffset=%zu dstByteOffset=%zu\n",
                          row, col, srcIdx, dstPixelOffset, dstByteOffset);

                if (task.type == mxUINT16_CLASS && srcIdx + 1 < rowBuffer.size()) {
                    uint16_T val16 = *reinterpret_cast<uint16_T*>(rowBuffer.data() + srcIdx);
                    mexPrintf("[DEBUG]     Read UINT16 value from src: %u\n", val16);
                } else if (task.type == mxUINT8_CLASS && srcIdx < rowBuffer.size()) {
                    uint8_T val8 = *(rowBuffer.data() + srcIdx);
                    mexPrintf("[DEBUG]     Read UINT8 value from src: %u\n", val8);
                }
            }

            if (task.type == mxUINT8_CLASS || task.type == mxUINT16_CLASS) {
                std::memcpy(static_cast<uint8_t*>(task.dst) + dstByteOffset,
                            rowBuffer.data() + srcIdx,
                            pixelSize);
            } else {
                mexErrMsgIdAndTxt("TIFFLoad:UnsupportedType", "Unsupported output data type.");
            }

            // Print written value
            if (row < 3 && col < 3) {
                if (task.type == mxUINT16_CLASS) {
                    uint16_T val_written = *reinterpret_cast<uint16_T*>(
                        static_cast<uint8_t*>(task.dst) + dstByteOffset);
                    mexPrintf("[DEBUG]     Written UINT16 value to dst: %u\n", val_written);
                } else if (task.type == mxUINT8_CLASS) {
                    uint8_T val_written = *(static_cast<uint8_t*>(task.dst) + dstByteOffset);
                    mexPrintf("[DEBUG]     Written UINT8 value to dst: %u\n", val_written);
                }
            }
        }
    }

    mexPrintf("[DEBUG] ====== LoadTask End ======\n\n");
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
    if ((uint32_t)(x + width) > imgWidth || (uint32_t)(y + height) > imgHeight)
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
