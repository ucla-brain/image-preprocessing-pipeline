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

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16)){
        TIFFClose(tif);
        mexErrMsgIdAndTxt("load_bl_tif:Type", "Only 8/16-bit grayscale TIFFs are supported");
    }

    uint32_t tileWidth = 0, tileHeight = 0;
    int isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    }

    const std::size_t bytesPerPixel = bitsPerSample / 8;
    uint8_t* outBase = static_cast<uint8_t*>(task.dstBase) + task.zIndex * task.pixelsPerSlice * bytesPerPixel;

    if (isTiled) {
        std::vector<uint8_t> tilebuf(TIFFTileSize(tif));
        for (int outCol = 0; outCol < task.roiW; ++outCol) {
            int imgCol = task.roiX + outCol;
            if (imgCol < 0 || imgCol >= static_cast<int>(imgWidth)) continue;
            for (int outRow = 0; outRow < task.roiH; ++outRow) {
                int imgRow = task.roiY + outRow;
                if (imgRow < 0 || imgRow >= static_cast<int>(imgHeight)) continue;

                uint32_t tileX = (imgCol / tileWidth) * tileWidth;
                uint32_t tileY = (imgRow / tileHeight) * tileHeight;
                tsize_t tileIdx = TIFFComputeTile(tif, imgCol, imgRow, 0, 0);

                if (TIFFReadEncodedTile(tif, tileIdx, tilebuf.data(), tilebuf.size()) < 0) {
                    TIFFClose(tif);
                    mexErrMsgIdAndTxt("load_bl_tif:ReadTile", "Failed reading tile at x=%u y=%u", tileX, tileY);
                }

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                    tsize_t tilesize = TIFFTileSize(tif);
                    size_t n_tile_pixels = tilesize / bytesPerPixel;
                    swap_uint16_buf(tilebuf.data(), n_tile_pixels);
                }

                uint32_t relY = imgRow - tileY;
                uint32_t relX = imgCol - tileX;
                std::size_t tileStride = tileWidth * bytesPerPixel;
                std::size_t offset = relY * tileStride + relX * bytesPerPixel;

                std::size_t dstIndex = outRow + outCol * task.roiH;
                std::size_t dstOffset = dstIndex * bytesPerPixel;

                std::memcpy(outBase + dstOffset, tilebuf.data() + offset, bytesPerPixel);
            }
        }
    } else {
        std::vector<uint8_t> scanline(imgWidth * bytesPerPixel);
        int lastScanlineRead = -1;
        for (int outCol = 0; outCol < task.roiW; ++outCol) {
            int imgCol = task.roiX + outCol;
            if (imgCol < 0 || imgCol >= static_cast<int>(imgWidth)) continue;
            for (int outRow = 0; outRow < task.roiH; ++outRow) {
                int imgRow = task.roiY + outRow;
                if (imgRow < 0 || imgRow >= static_cast<int>(imgHeight)) continue;

                if (lastScanlineRead != imgRow) {
                    if (!TIFFReadScanline(tif, scanline.data(), imgRow)) {
                        TIFFClose(tif);
                        mexErrMsgIdAndTxt("load_bl_tif:Read", "Read row %d failed", imgRow);
                    }
                    if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                        swap_uint16_buf(scanline.data(), imgWidth);
                    }
                    lastScanlineRead = imgRow;
                }

                std::size_t srcOffset = imgCol * bytesPerPixel;
                std::size_t dstIndex = outRow + outCol * task.roiH;
                std::size_t dstOffset = dstIndex * bytesPerPixel;
                std::memcpy(outBase + dstOffset, scanline.data() + srcOffset, bytesPerPixel);
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

    // Create MATLAB output array [height, width, numSlices] (column-major)
    mwSize dims[3] = { static_cast<mwSize>(roiH),
                       static_cast<mwSize>(roiW),
                       static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
    void* outData = mxGetData(plhs[0]);
    std::size_t pixelsPerSlice = static_cast<std::size_t>(roiH) * roiW;

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
    // For debugging: print first 10 values
    uint16_t* arr = static_cast<uint16_t*>(outData);
    printf("First 10 values: ");
    for (int i = 0; i < 10; ++i) printf("%d ", arr[i]);
    printf("\n");
}
