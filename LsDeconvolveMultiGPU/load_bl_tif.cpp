#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>   // memcpy

typedef unsigned char  uint8_T;
typedef unsigned short uint16_T;

struct LoadTask
{
    std::string filename;
    int  y, x, height, width;
    size_t zindex;
    void* dst;
    size_t planeStride;      // width * height  (bytes per plane / pixelSize)
    mxClassID type;
};

// ------------------------------------------------------------
//  Copy one TIFF file's sub-region into the MATLAB output
// ------------------------------------------------------------
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
    if ((uint32_t)(task.x + task.width - 1) > imgWidth || (uint32_t)(task.y + task.height - 1) > imgHeight)
        mexErrMsgIdAndTxt("TIFFLoad:SubregionBounds", "Subregion out of bounds in: %s", task.filename.c_str());

    size_t pixelSize = bitsPerSample / 8;
    if (pixelSize != 1 && pixelSize != 2)
        mexErrMsgIdAndTxt("TIFFLoad:InvalidPixelSize", "Unsupported pixel size: %zu", pixelSize);

    size_t scanlineSize = imgWidth * pixelSize;
    std::vector<uint8_t> rowBuffer(scanlineSize);

    for (int row = 0; row < task.height; ++row) {
        if (!TIFFReadScanline(tif, rowBuffer.data(), task.y - 1 + row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline in: %s", task.filename.c_str());

        for (int col = 0; col < task.width; ++col) {
            size_t srcIdx = static_cast<size_t>(task.x - 1 + col) * pixelSize; // <-- FIXED!
            size_t dstPixelOffset = static_cast<size_t>(row) +
                                    static_cast<size_t>(col) * task.height +
                                    task.zindex * task.planeStride;
            size_t dstByteOffset = dstPixelOffset * pixelSize;

            if (task.type == mxUINT8_CLASS || task.type == mxUINT16_CLASS) {
                std::memcpy(static_cast<uint8_t*>(task.dst) + dstByteOffset,
                            rowBuffer.data() + srcIdx,
                            pixelSize);
            } else {
                mexErrMsgIdAndTxt("TIFFLoad:UnsupportedType", "Unsupported output data type.");
            }
        }
    }
    TIFFClose(tif);
}

// ------------------------------------------------------------
//  MEX gateway
// ------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5)
        mexErrMsgIdAndTxt("TIFFLoad:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width [, num_threads])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("TIFFLoad:InvalidInput",
            "First argument must be a cell array of filenames.");

    // ---------- gather filenames ----------
    const size_t numSlices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> files(numSlices);
    for (size_t i = 0; i < numSlices; ++i)
    {
        if (!mxIsChar(mxGetCell(prhs[0], i)))
            mexErrMsgIdAndTxt("TIFFLoad:InvalidCell",
                              "File list must contain strings (index %zu).", i);
        char* s = mxArrayToString(mxGetCell(prhs[0], i));
        if (!s || !*s)
            mexErrMsgIdAndTxt("TIFFLoad:EmptyPath",
                              "Empty filename at index %zu.", i);
        files[i] = s;  mxFree(s);
    }

    // ---------- region parameters ----------
    const int y      = (int)mxGetScalar(prhs[1]);
    const int x      = (int)mxGetScalar(prhs[2]);
    const int height = (int)mxGetScalar(prhs[3]);
    const int width  = (int)mxGetScalar(prhs[4]);

    // ---------- read first file for metadata ----------
    TIFF* tif = TIFFOpen(files[0].c_str(), "r");
    if (!tif) mexErrMsgIdAndTxt("TIFFLoad:OpenFail",
                                "Failed to open: %s", files[0].c_str());

    uint32_t imgW, imgH;
    uint16_t bps, spp = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFClose(tif);

    if (spp != 1)
        mexErrMsgIdAndTxt("TIFFLoad:GrayscaleOnly",
                          "Only grayscale TIFFs are supported.");
    if (bps != 8 && bps != 16)
        mexErrMsgIdAndTxt("TIFFLoad:Depth",
                          "Only 8- or 16-bit TIFFs are supported.");
    if ((uint32_t)(x + width)  > imgW ||
        (uint32_t)(y + height) > imgH)
        mexErrMsgIdAndTxt("TIFFLoad:Bounds",
                          "Requested subregion is out of image bounds.");

    // ---------- create MATLAB output ----------
    mxClassID cls = (bps == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    mwSize dims[3] = { (mwSize)width, (mwSize)height, (mwSize)numSlices };
    plhs[0] = mxCreateNumericArray(3, dims, cls, mxREAL);

    void*  outData     = mxGetData(plhs[0]);
    size_t planeStride = (size_t)width * height;   // pixels per slice

    // ---------- process each slice (single-thread for now) ----------
    for (size_t z = 0; z < files.size(); ++z)
    {
        LoadTask t { files[z], y, x, height, width,
                     z, outData, planeStride, cls };
        load_subregion(t);
    }
}
