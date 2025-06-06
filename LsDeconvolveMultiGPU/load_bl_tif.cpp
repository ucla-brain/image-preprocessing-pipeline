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
static void load_subregion (const LoadTask& task)
{
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("TIFFLoad:OpenFail",
                          "Failed to open: %s", task.filename.c_str());

    uint32_t imgW, imgH;
    uint16_t bps, spp = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);

    if (spp != 1)
        mexErrMsgIdAndTxt("TIFFLoad:NotGrayscale",
                          "Only grayscale TIFFs are supported: %s",
                          task.filename.c_str());
    if (bps != 8 && bps != 16)
        mexErrMsgIdAndTxt("TIFFLoad:UnsupportedDepth",
                          "Only 8- or 16-bit TIFFs are supported.");
    if ((uint32_t)(task.x + task.width ) > imgW ||
        (uint32_t)(task.y + task.height) > imgH)
        mexErrMsgIdAndTxt("TIFFLoad:SubregionBounds",
                          "Subregion out of bounds in: %s",
                          task.filename.c_str());

    const size_t pixelSize = bps / 8;          // 1 or 2
    const size_t scanline  = imgW * pixelSize; // bytes in one TIFF row
    std::vector<uint8_t> rowBuf(scanline);

    for (int row = 0; row < task.height; ++row)
    {
        if (!TIFFReadScanline(tif, rowBuf.data(), task.y + row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError",
                              "Failed to read scanline in: %s",
                              task.filename.c_str());

        for (int col = 0; col < task.width; ++col)
        {
            const size_t srcIdx = static_cast<size_t>(task.x + col) * pixelSize;

            // === column-major linear index ===
            size_t dstPixel =  static_cast<size_t>(row)                /* y   */
                             + static_cast<size_t>(col) * task.height  /* x   */
                             + task.zindex * task.planeStride;         /* z   */
            size_t dstByte  = dstPixel * pixelSize;

            std::memcpy( static_cast<uint8_t*>(task.dst) + dstByte,
                         rowBuf.data() + srcIdx,
                         pixelSize );
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
