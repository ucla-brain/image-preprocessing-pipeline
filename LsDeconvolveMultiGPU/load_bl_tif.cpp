#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>

/* --------------------------------------------------------------------- */
struct LoadTask
{
    std::string filename;
    int  roiY, roiX, roiH, roiW;    // All 0-based
    std::size_t zIndex;             // 0-based
    void*       dstBase;
    std::size_t pixelsPerSlice;
    mxClassID   matlabType;
};

/* --------------------------------------------------------------------- */
// Helper: throws on error
static void throw_mex(const char* id, const char* fmt, const char* s) {
    char msg[512];
    snprintf(msg, sizeof(msg), fmt, s);
    mexErrMsgIdAndTxt(id, "%s", msg);
}

// Generic subregion extraction for both tiled and scanline/strip TIFFs
static void copySubRegion(const LoadTask& task)
{
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if (!tif)
        throw_mex("load_bl_tif:OpenFail", "Cannot open %s", task.filename.c_str());

    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type", "Only 8/16-bit grayscale TIFFs are supported");

    const std::size_t bytesPerPixel = bitsPerSample / 8;

    // Detect tiled TIFF
    uint32_t tileWidth = 0, tileHeight = 0;
    int isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    }

    // MATLAB is column-major, with explicit transpose: output is [width, height, z]
    // dstIndex = row + col*roiH + zIndex*pixelsPerSlice

    if (isTiled) {
        // -------- TILED TIFF --------
        std::vector<uint8_t> tilebuf(TIFFTileSize(tif));
        for (int row = 0; row < task.roiH; ++row) {
            uint32_t imgY = task.roiY + row;
            for (int col = 0; col < task.roiW; ++col) {
                uint32_t imgX = task.roiX + col;

                // Determine which tile contains this pixel
                uint32_t tileX = (imgX / tileWidth) * tileWidth;
                uint32_t tileY = (imgY / tileHeight) * tileHeight;
                tsize_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                // Read tile (cache last if you want to optimize further)
                if (TIFFReadEncodedTile(tif, tileIdx, tilebuf.data(), tilebuf.size()) < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:ReadTile", "Failed reading tile at x=%u y=%u", tileX, tileY);

                // Offset of this pixel in the tile buffer
                uint32_t relY = imgY - tileY;
                uint32_t relX = imgX - tileX;
                std::size_t tileStride = tileWidth * bytesPerPixel;
                std::size_t offset = relY * tileStride + relX * bytesPerPixel;

                // Write pixel to output
                std::size_t dstIndex = row + col * task.roiH + task.zIndex * task.pixelsPerSlice;
                std::size_t dstOffset = dstIndex * bytesPerPixel;

                std::memcpy(static_cast<uint8_t*>(task.dstBase) + dstOffset,
                            tilebuf.data() + offset, bytesPerPixel);
            }
        }
    } else {
        // -------- STRIPPED/SCANLINE TIFF --------
        // Use TIFFReadScanline for each image row, then memcpy subregion
        std::vector<uint8_t> scanline(imgWidth * bytesPerPixel);
        for (int row = 0; row < task.roiH; ++row) {
            uint32_t tifRow = static_cast<uint32_t>(task.roiY + row);
            if (!TIFFReadScanline(tif, scanline.data(), tifRow))
                mexErrMsgIdAndTxt("load_bl_tif:Read", "Read row %u failed", tifRow);

            for (int col = 0; col < task.roiW; ++col) {
                std::size_t srcPixel = static_cast<std::size_t>(task.roiX + col);
                std::size_t srcOffset = srcPixel * bytesPerPixel;

                std::size_t dstIndex = row + col * task.roiH + task.zIndex * task.pixelsPerSlice;
                std::size_t dstOffset = dstIndex * bytesPerPixel;

                std::memcpy(static_cast<uint8_t*>(task.dstBase) + dstOffset,
                            scanline.data() + srcOffset, bytesPerPixel);
            }
        }
    }
    TIFFClose(tif);
}

/* --------------------------------------------------------------------- */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width)");

    // File list (cell array of strings)
    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First argument must be a cell array of filenames");

    const std::size_t numSlices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> fileList(numSlices);
    for (std::size_t i = 0; i < numSlices; ++i)
    {
        char* s = mxArrayToString(mxGetCell(prhs[0], i));
        fileList[i] = s;
        mxFree(s);
    }

    // Consistent: parse all ROI and z indices as 0-based (MATLAB gives 1-based)
    const int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1; // 0-based
    const int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1; // 0-based
    const int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW  = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI parameters invalid");

    // Check bit-depth from first file
    TIFF* tif0 = TIFFOpen(fileList[0].c_str(), "r");
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open %s", fileList[0].c_str());
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif0, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif0, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFClose(tif0);
    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type","Only 8/16-bit grayscale TIFFs are supported");

    const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const std::size_t bytesPerPixel = bitsPerSample / 8;

    // Create MATLAB output array [width, height, numSlices] (column-major)
    mwSize dims[3] = { static_cast<mwSize>(roiW),
                       static_cast<mwSize>(roiH),
                       static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    std::size_t pixelsPerSlice = static_cast<std::size_t>(roiW) * roiH;

    // Loop over all Z (all 0-based now!)
    for (std::size_t z = 0; z < numSlices; ++z)
    {
        LoadTask task;
        task.filename       = fileList[z];
        task.roiY           = roiY0; // all 0-based now
        task.roiX           = roiX0;
        task.roiH           = roiH;
        task.roiW           = roiW;
        task.zIndex         = z;
        task.dstBase        = outData;
        task.pixelsPerSlice = pixelsPerSlice;
        task.matlabType     = outType;
        copySubRegion(task);
    }
}
