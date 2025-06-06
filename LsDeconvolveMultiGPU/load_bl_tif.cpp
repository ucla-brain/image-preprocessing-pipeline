#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>

// Helper: read a subregion from a scanline TIFF
void read_scanline_tiff(
    TIFF* tif, int x, int y, int width, int height, void* dst, size_t planeStride, mxClassID type)
{
    uint16_t bitsPerSample;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    size_t pixelSize = bitsPerSample / 8;
    uint32_t imgWidth;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    std::vector<uint8_t> rowBuffer(imgWidth * pixelSize);

    for (int row = 0; row < height; ++row) {
        if (!TIFFReadScanline(tif, rowBuffer.data(), y + row))
            mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read scanline");

        for (int col = 0; col < width; ++col) {
            size_t srcIdx = (x + col) * pixelSize;
            size_t dstPixelOffset = row + col * height;
            size_t dstByteOffset = dstPixelOffset * pixelSize;

            std::memcpy(static_cast<uint8_t*>(dst) + dstByteOffset, rowBuffer.data() + srcIdx, pixelSize);
        }
    }
}

// Helper: read a subregion from a tiled TIFF
void read_tiled_tiff(
    TIFF* tif, int x, int y, int width, int height, void* dst, size_t planeStride, mxClassID type)
{
    uint32_t tileWidth, tileLength;
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileLength);
    uint16_t bitsPerSample;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    size_t pixelSize = bitsPerSample / 8;
    uint32_t imgWidth, imgHeight;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);

    std::vector<uint8_t> tileBuffer(TIFFTileSize(tif));

    for (int y_out = 0; y_out < height; ++y_out) {
        int y_img = y + y_out;
        int tileY = (y_img / tileLength) * tileLength;
        int y_in_tile = y_img % tileLength;
        for (int x_out = 0; x_out < width; ++x_out) {
            int x_img = x + x_out;
            int tileX = (x_img / tileWidth) * tileWidth;
            int x_in_tile = x_img % tileWidth;

            // Read tile if it's the first pixel in this row and column block
            tsize_t tileIdx = TIFFComputeTile(tif, tileX, tileY, 0, 0);
            static tsize_t lastTileIdx = -1;
            static std::vector<uint8_t> lastTile;
            if (lastTileIdx != tileIdx) {
                if (TIFFReadEncodedTile(tif, tileIdx, tileBuffer.data(), tileBuffer.size()) == -1)
                    mexErrMsgIdAndTxt("TIFFLoad:ReadError", "Failed to read tile");
                lastTile = tileBuffer;
                lastTileIdx = tileIdx;
            }

            size_t tilePixelOffset = (y_in_tile * tileWidth + x_in_tile) * pixelSize;
            size_t dstPixelOffset = y_out + x_out * height;
            size_t dstByteOffset = dstPixelOffset * pixelSize;

            std::memcpy(static_cast<uint8_t*>(dst) + dstByteOffset, lastTile.data() + tilePixelOffset, pixelSize);
        }
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5)
        mexErrMsgIdAndTxt("TIFFLoad:Usage", "Usage: img = load_bl_tif(files, y, x, height, width)");

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

    // Validate metadata of the first image
    TIFF* tif = TIFFOpen(filenames[0].c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("TIFFLoad:OpenFail", "Failed to open: %s", filenames[0].c_str());

    uint32_t imgWidth, imgHeight;
    uint16_t bitsPerSample, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    int isTiled = TIFFIsTiled(tif);

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
        TIFF* tif = TIFFOpen(filenames[i].c_str(), "r");
        if (!tif)
            mexErrMsgIdAndTxt("TIFFLoad:OpenFail", "Failed to open: %s", filenames[i].c_str());
        void* dst = static_cast<uint8_t*>(outData) + i * planeStride * (bitsPerSample / 8);
        if (isTiled)
            read_tiled_tiff(tif, x, y, width, height, dst, planeStride, outType);
        else
            read_scanline_tiff(tif, x, y, width, height, dst, planeStride, outType);
        TIFFClose(tif);
    }
}
