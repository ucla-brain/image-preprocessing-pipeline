/**
 * load_bl_tif.cpp
 * ---------------------------------------------------------------------------
 * Read a rectangular sub-region from a stack of single-channel TIFF files and
 * return it to MATLAB with dimensions  [width  height  depth] (column-major).
 * The code is purposely single-threaded for clarity.
 * ---------------------------------------------------------------------------
 *  mex -largeArrayDims CXXFLAGS="$CXXFLAGS -std=c++17" load_bl_tif.cpp -ltiff
 */

#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>  // memcpy

// Enable to see the first few rows/cols copied
#define LOAD_BL_DEBUG 0

/* ------------------------------------------------------------------------- */
/*  Small aliases                                                            */
/* ------------------------------------------------------------------------- */

using uint8  = unsigned char;
using uint16 = unsigned short;

/* ------------------------------------------------------------------------- */
/*  One “task” == one TIFF file / one Z-slice                                */
/* ------------------------------------------------------------------------- */
struct LoadTask
{
    std::string filename;

    // user-requested rectangle in the TIFF (1-based)*:
    int roiY;      // top-left y
    int roiX;      // top-left x
    int roiH;      // height
    int roiW;      // width

    std::size_t zIndex;        // this TIFF→ depth slice in output
    void*       dstBase;       // MATLAB data pointer
    std::size_t pixelsPerPlane; // width * height
    mxClassID   matlabType;     // mxUINT8_CLASS or mxUINT16_CLASS
};

/* ------------------------------------------------------------------------- */
/*  Helper that fills one Z-slice                                            */
/* ------------------------------------------------------------------------- */
static void copySubRegion( const LoadTask& task )
{
    /* ---------------- Open & validate TIFF ---------------- */
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
                          "Cannot open file: %s", task.filename.c_str());

    uint32_t imgW  = 0, imgH = 0;
    uint16   bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1)
        mexErrMsgIdAndTxt("load_bl_tif:RGB",
                          "Only single-channel TIFFs are supported: %s",
                          task.filename.c_str());

    if (bitsPerSample != 8 && bitsPerSample != 16)
        mexErrMsgIdAndTxt("load_bl_tif:BitDepth",
                          "Only 8- or 16-bit TIFFs are supported.");

    if (static_cast<uint32_t>(task.roiX - 1 + task.roiW) > imgW ||
        static_cast<uint32_t>(task.roiY - 1 + task.roiH) > imgH)
        mexErrMsgIdAndTxt("load_bl_tif:Bounds",
                          "Requested sub-region is outside the image: %s",
                          task.filename.c_str());

    const std::size_t bytesPerPixel = bitsPerSample / 8;
    const std::size_t scanlineBytes = static_cast<std::size_t>(imgW) * bytesPerPixel;

    std::vector<uint8> scanline(scanlineBytes);

    /* ---------------- Copy row-by-row into MATLAB array ---------------- */
    for (int rowInRoi = 0; rowInRoi < task.roiH; ++rowInRoi)
    {
        // TIFF rows are 0-based:
        const uint32_t tifRow = static_cast<uint32_t>(task.roiY - 1 + rowInRoi);

        if (!TIFFReadScanline(tif, scanline.data(), tifRow))
            mexErrMsgIdAndTxt("load_bl_tif:ReadError",
                              "Failed reading row %u from '%s'",
                              tifRow, task.filename.c_str());

        for (int colInRoi = 0; colInRoi < task.roiW; ++colInRoi)
        {
            /* -------- Source address inside this scanline -------- */
            const std::size_t srcPixelIndex = static_cast<std::size_t>(task.roiX - 1 + colInRoi);
            const std::size_t srcByteOffset = srcPixelIndex * bytesPerPixel;

            /* -------- Destination address in MATLAB array --------
             *
             *  MATLAB is column-major: fastest index is along *columns* (our X).
             *  Output dims = [width height depth]
             *
             *  linear index =  x
             *                + y * width
             *                + z * (width * height)
             */
            const std::size_t dstPixelIndex =
                    static_cast<std::size_t>(colInRoi)                              // x
                  + static_cast<std::size_t>(rowInRoi) * task.roiW                  // y
                  + task.zIndex * task.pixelsPerPlane;                              // z
            const std::size_t dstByteOffset = dstPixelIndex * bytesPerPixel;

            std::memcpy( static_cast<uint8*>(task.dstBase) + dstByteOffset,
                         scanline.data() + srcByteOffset,
                         bytesPerPixel );
        }
    }

    TIFFClose(tif);
}

/* ------------------------------------------------------------------------- */
/*  mexFunction                                                              */
/* ------------------------------------------------------------------------- */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width)");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First argument must be a cell array of filenames.");

    /* ---------------- Parse input file list ---------------- */
    const std::size_t depth = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> filenames(depth);

    for (std::size_t i = 0; i < depth; ++i)
    {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell))
            mexErrMsgIdAndTxt("load_bl_tif:FileName",
                              "File list cell %zu is not a string", i + 1);

        char* tmp = mxArrayToString(cell);
        filenames[i] = tmp;
        mxFree(tmp);
    }

    /* ---------------- ROI parameters (1-based) ---------------- */
    const int roiY = static_cast<int>(mxGetScalar(prhs[1]));
    const int roiX = static_cast<int>(mxGetScalar(prhs[2]));
    const int roiH = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW = static_cast<int>(mxGetScalar(prhs[4]));

    if (roiY < 1 || roiX < 1 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI",
                          "ROI parameters must be positive integers.");

    /* ---------------- Inspect first image to know type ---------------- */
    TIFF* tif0 = TIFFOpen(filenames[0].c_str(), "r");
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
                          "Cannot open first file: %s", filenames[0].c_str());

    uint16 bitsPerSample = 0;  uint16 samplesPerPixel = 1;
    TIFFGetField(tif0, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif0, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFClose(tif0);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
                          "Only 8/16-bit single-channel TIFFs are supported.");

    const mxClassID matlabType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const std::size_t bytesPerPixel = bitsPerSample / 8;

    /* ---------------- Allocate MATLAB output ---------------- */
    mwSize dims[3] = { static_cast<mwSize>(roiW),
                       static_cast<mwSize>(roiH),
                       static_cast<mwSize>(depth) };
    plhs[0] = mxCreateNumericArray(3, dims, matlabType, mxREAL);
    void* outPtr = mxGetData(plhs[0]);
    const std::size_t pixelsPerPlane = static_cast<std::size_t>(roiW) * roiH;

    /* ---------------- Copy each slice ---------------- */
    for (std::size_t z = 0; z < depth; ++z)
    {
        LoadTask task{ filenames[z], roiY, roiX, roiH, roiW,
                       z, outPtr, pixelsPerPlane, matlabType };

#if LOAD_BL_DEBUG
        mexPrintf("[DEBUG] Copying Z=%zu from '%s'\n",
                  z, task.filename.c_str());
#endif
        copySubRegion(task);
    }
}
