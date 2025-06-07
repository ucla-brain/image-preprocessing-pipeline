#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

struct LoadTask
{
    std::string filename;
    int roiY, roiX, roiH, roiW;    // All 0-based
    std::size_t zIndex;            // 0-based
    void* dstBase;
    std::size_t pixelsPerSlice;
    mxClassID matlabType;
    int cropH, cropW;              // cropped (actual) block size
    int imgW, imgH;                // full image size
};

/* Detect endianness */
static bool is_system_little_endian() {
    uint16_t x = 1;
    return *((uint8_t*)&x) == 1;
}

static void swap_uint16_buf(void* buf, size_t count) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < count; ++i) {
        uint16_t v = p[i];
        p[i] = (v >> 8) | (v << 8);
    }
}

static void throw_mex(const char* id, const char* fmt, const char* s) {
    char msg[512];
    snprintf(msg, sizeof(msg), fmt, s);
    mexErrMsgIdAndTxt(id, "%s", msg);
}

/* -------- Generic subregion extraction for both tiled and scanline TIFFs -------- */
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

    const std::size_t bytesPerPixel = bitsPerSample / 8;

    uint32_t tileWidth = 0, tileHeight = 0;
    int isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    }

    // For debug: print ROI
    mexPrintf("  [MEX] File: %s | ImageSize: [%u,%u] | ROI: y=%d..%d, x=%d..%d | BlockSize: [%d,%d] | Crop: [%d,%d] | Z=%zu\n",
        task.filename.c_str(), imgHeight, imgWidth,
        task.roiY + 1, task.roiY + task.cropH,
        task.roiX + 1, task.roiX + task.cropW,
        task.roiH, task.roiW, task.cropH, task.cropW, task.zIndex + 1);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type", "Only 8/16-bit grayscale TIFFs are supported");

    // Output is [height, width, z] (column-major): dstIndex = row + col*cropH + z*pixelsPerSlice

    if (isTiled) {
        std::vector<uint8_t> tilebuf(TIFFTileSize(tif));
        for (int row = 0; row < task.cropH; ++row) {
            uint32_t imgY = task.roiY + row;
            for (int col = 0; col < task.cropW; ++col) {
                uint32_t imgX = task.roiX + col;

                uint32_t tileX = (imgX / tileWidth) * tileWidth;
                uint32_t tileY = (imgY / tileHeight) * tileHeight;
                tsize_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (TIFFReadEncodedTile(tif, tileIdx, tilebuf.data(), tilebuf.size()) < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:ReadTile", "Failed reading tile at x=%u y=%u", tileX, tileY);

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                    tsize_t tilesize = TIFFTileSize(tif);
                    size_t n_tile_pixels = tilesize / bytesPerPixel;
                    swap_uint16_buf(tilebuf.data(), n_tile_pixels);
                }

                uint32_t relY = imgY - tileY;
                uint32_t relX = imgX - tileX;
                std::size_t tileStride = tileWidth * bytesPerPixel;
                std::size_t offset = relY * tileStride + relX * bytesPerPixel;

                std::size_t dstIndex = row + col * task.cropH + task.zIndex * task.pixelsPerSlice;
                std::size_t dstOffset = dstIndex * bytesPerPixel;

                std::memcpy(static_cast<uint8_t*>(task.dstBase) + dstOffset,
                            tilebuf.data() + offset, bytesPerPixel);
            }
        }
    } else {
        std::vector<uint8_t> scanline(imgWidth * bytesPerPixel);
        for (int row = 0; row < task.cropH; ++row) {
            uint32_t tifRow = static_cast<uint32_t>(task.roiY + row);
            if (!TIFFReadScanline(tif, scanline.data(), tifRow))
                mexErrMsgIdAndTxt("load_bl_tif:Read", "Read row %u failed", tifRow);

            if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                swap_uint16_buf(scanline.data(), imgWidth);
            }

            for (int col = 0; col < task.cropW; ++col) {
                std::size_t srcPixel = static_cast<std::size_t>(task.roiX + col);
                std::size_t srcOffset = srcPixel * bytesPerPixel;

                std::size_t dstIndex = row + col * task.cropH + task.zIndex * task.pixelsPerSlice;
                std::size_t dstOffset = dstIndex * bytesPerPixel;

                std::memcpy(static_cast<uint8_t*>(task.dstBase) + dstOffset,
                            scanline.data() + srcOffset, bytesPerPixel);
            }
        }
    }
    TIFFClose(tif);
}

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width)");

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

    int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    int roiW  = static_cast<int>(mxGetScalar(prhs[4]));

    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI parameters invalid");

    // Get image size from first file for crop logic
    TIFF* tif0 = TIFFOpen(fileList[0].c_str(), "r");
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open %s", fileList[0].c_str());
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif0, TIFFTAG_IMAGEWIDTH , &imgWidth);
    TIFFGetField(tif0, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif0, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif0, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFClose(tif0);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type","Only 8/16-bit grayscale TIFFs are supported");

    // Crop at edge if needed
    int cropH = roiH;
    int cropW = roiW;
    if (roiY0 + roiH > (int)imgHeight)
        cropH = std::max(0, (int)imgHeight - roiY0);
    if (roiX0 + roiW > (int)imgWidth)
        cropW = std::max(0, (int)imgWidth - roiX0);

    const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    const std::size_t bytesPerPixel = bitsPerSample / 8;

    // Output array: [height, width, numSlices]
    mwSize dims[3] = { static_cast<mwSize>(roiH),
                       static_cast<mwSize>(roiW),
                       static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    std::size_t pixelsPerSlice = static_cast<std::size_t>(roiH) * roiW;

    // Zero the output buffer
    memset(outData, 0, pixelsPerSlice * numSlices * bytesPerPixel);

    // Loop over Z
    for (std::size_t z = 0; z < numSlices; ++z)
    {
        TIFF* tifa = TIFFOpen(fileList[z].c_str(), "r");
        if (!tifa)
            mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open %s", fileList[z].c_str());
        uint32_t imgWa = 0, imgHa = 0;
        TIFFGetField(tifa, TIFFTAG_IMAGEWIDTH , &imgWa);
        TIFFGetField(tifa, TIFFTAG_IMAGELENGTH, &imgHa);
        TIFFClose(tifa);
        int cropHz = roiH, cropWz = roiW;
        if (roiY0 + roiH > (int)imgHa)
            cropHz = std::max(0, (int)imgHa - roiY0);
        if (roiX0 + roiW > (int)imgWa)
            cropWz = std::max(0, (int)imgWa - roiX0);

        // If cropHz <= 0 or cropWz <= 0, skip (output remains zero)
        if (cropHz <= 0 || cropWz <= 0)
            continue;

        // Only copy for the valid portion
        LoadTask task;
        task.filename       = fileList[z];
        task.roiY           = roiY0;
        task.roiX           = roiX0;
        task.roiH           = roiH;
        task.roiW           = roiW;
        task.cropH          = cropHz;
        task.cropW          = cropWz;
        task.imgH           = imgHa;
        task.imgW           = imgWa;
        task.zIndex         = z;
        task.dstBase        = outData;
        task.pixelsPerSlice = pixelsPerSlice;
        task.matlabType     = outType;
        copySubRegion(task);
    }
    // Print first 10 values for debug
    if (outType == mxUINT16_CLASS) {
        uint16_t* arr = static_cast<uint16_t*>(outData);
        mexPrintf("First 10 values: ");
        for (int i = 0; i < 10; ++i) mexPrintf("%d ", arr[i]);
        mexPrintf("\n");
    } else {
        uint8_t* arr = static_cast<uint8_t*>(outData);
        mexPrintf("First 10 values: ");
        for (int i = 0; i < 10; ++i) mexPrintf("%d ", arr[i]);
        mexPrintf("\n");
    }
}
