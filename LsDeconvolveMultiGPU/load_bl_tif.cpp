#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <memory>

// --- Struct with new offset fields ---
constexpr uint16_t kSupportedBitDepth8  = 8;
constexpr uint16_t kSupportedBitDepth16 = 16;

// RAII wrapper for mxArrayToUTF8String()
struct MatlabString {
private:
    char* ptr;

public:
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString", "Failed to convert string from mxArray");
    }

    ~MatlabString() { mxFree(ptr); }

    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

struct LoadTask
{
    const std::size_t in_row0, in_col0;     // Where to start reading in TIFF image (Y, X)
    const std::size_t out_row0, out_col0;   // Where to write in output buffer (Y, X)
    const std::size_t cropH, cropW;         // How many rows and columns to copy
    const std::size_t roiH, roiW;           // Full output block size
    const std::size_t zIndex;       // 0-based Z index in output buffer
    void* dstBase;
    const std::size_t pixelsPerSlice;
    const std::string path;
    const bool transpose;

    LoadTask(
        std::size_t inY, std::size_t inX, std::size_t outY, std::size_t outX,
        std::size_t h, std::size_t w, std::size_t roiH_, std::size_t roiW_,
        std::size_t z, void* dst, std::size_t pps, std::string filename, bool transpose_
    ) :
        in_row0(inY), in_col0(inX), out_row0(outY), out_col0(outX),
        cropH(h), cropW(w), roiH(roiH_), roiW(roiW_),
        zIndex(z), dstBase(dst), pixelsPerSlice(pps), path(filename), transpose(transpose_){}
};

struct TiffCloser {
    void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

inline void getImageSize(TIFF* tif, uint32_t& w, uint32_t& h) {
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
}

inline std::size_t computeDstIndex(const LoadTask& task, int row, int col) noexcept {
    if (!task.transpose) {
        return (task.out_row0 + row)
             + (task.out_col0 + col) * task.roiH
             + task.zIndex * task.pixelsPerSlice;
    } else {
        return (task.out_col0 + col)
             + (task.out_row0 + row) * task.roiW
             + task.zIndex * task.pixelsPerSlice;
    }
}


// --- Byte swap helper ---
static void swap_uint16_buf(void* buf, size_t count) {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < count; ++i) {
        uint16_t v = p[i];
        p[i] = (v >> 8) | (v << 8);
    }
}

// --- Read and copy a cropped subregion from a TIFF slice ---
static void copySubRegion(const LoadTask& task, TIFF* tif, uint8_t bytesPerPixel)
{
    uint32_t imgWidth, imgHeight;
    getImageSize(tif, imgWidth, imgHeight);
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    uint32_t tileWidth = 0, tileHeight = 0;
    int isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
    }
    if (samplesPerPixel != 1 || (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
        mexErrMsgIdAndTxt("load_bl_tif:Type", "Only 8/16-bit grayscale TIFFs are supported");

    if (isTiled) {
        tsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0)
            mexErrMsgIdAndTxt("load_bl_tif:TileSize", "Invalid tile size returned for file: %s", task.path.c_str());
        if (tileWidth == 0 || tileHeight == 0)
            mexErrMsgIdAndTxt("load_bl_tif:TileSize", "Invalid tile dimensions in TIFF metadata.");
        std::vector<uint8_t> tilebuf(tileSize);
        for (int row = 0; row < task.cropH; ++row) {
            uint32_t imgY = static_cast<uint32_t>(task.in_row0 + row);
            for (int col = 0; col < task.cropW; ++col) {
                uint32_t imgX = static_cast<uint32_t>(task.in_col0 + col);

                uint32_t tileX = (imgX / tileWidth) * tileWidth;
                uint32_t tileY = (imgY / tileHeight) * tileHeight;
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (TIFFReadEncodedTile(tif, tileIdx, tilebuf.data(), tilebuf.size()) < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:ReadTile", "Failed reading tile at x=%u y=%u", tileX, tileY);

                if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                    size_t n_tile_pixels = tileSize / bytesPerPixel;
                    swap_uint16_buf(tilebuf.data(), n_tile_pixels);
                }

                uint32_t relY = imgY - tileY;
                uint32_t relX = imgX - tileX;
                std::size_t rowStride = tileWidth * bytesPerPixel; // Bytes per row in tile
                std::size_t offset = relY * rowStride + relX * bytesPerPixel;

                // MATLAB column-major index: row + col * roiH + z * roiH * roiW
                std::size_t dstOffset = computeDstIndex(task, row, col) * bytesPerPixel;
                std::memcpy(
                    static_cast<uint8_t*>(task.dstBase) + dstOffset,
                    tilebuf.data() + offset,
                    bytesPerPixel
                );
            }
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;  // Default is whole image

        // Validate stripBufSize won't overflow
        size_t stripBufSize = static_cast<size_t>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (stripBufSize == 0 || stripBufSize > static_cast<size_t>(1) << 30) {
            mexErrMsgIdAndTxt("load_bl_tif:Memory", "Invalid or too large strip buffer size: %zu", stripBufSize);
        }
        std::vector<uint8_t> stripbuf(stripBufSize);
        tstrip_t currentStrip = (tstrip_t)-1;

        for (std::size_t row = 0; row < task.cropH; ++row) {
            uint32_t tifRow = static_cast<uint32_t>(task.in_row0 + row);
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);

            // Read strip if not already loaded
            if (stripIdx != currentStrip) {
                tsize_t nbytes = TIFFReadEncodedStrip(tif, stripIdx, stripbuf.data(), stripBufSize);
                if (nbytes < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:copySubRegion:ReadStrip",
                                      "TIFFReadEncodedStrip failed at strip %zu (row %u, file: %s)",
                                      stripIdx, tifRow, task.path.c_str());
                currentStrip = stripIdx;
            }

            // Row offset within strip
            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow = tifRow - stripStartRow;

            uint8_t* scanlinePtr = stripbuf.data() + (relRow * imgWidth * bytesPerPixel);

            // Endianness fix for 16-bit
            if (bytesPerPixel == 2 && TIFFIsByteSwapped(tif)) {
                swap_uint16_buf(scanlinePtr, imgWidth);
            }

            // Copy desired columns
            for (std::size_t col = 0; col < task.cropW; ++col) {
                std::size_t srcOffset = static_cast<std::size_t>(task.in_col0 + col) * bytesPerPixel;
                std::size_t dstOffset = computeDstIndex(task, row, col) * bytesPerPixel;
                std::memcpy(
                    static_cast<uint8_t*>(task.dstBase) + dstOffset,
                    scanlinePtr + srcOffset,
                    bytesPerPixel
                );
            }
        }
    }
}

// --- Main Entry Point ---
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    // Expected usage: img = load_bl_tif(filelist, y, x, height, width, transposeFlag)
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First argument must be a cell array of filenames");

    bool transpose = false;
    if (nrhs == 6) {
        if (!mxIsLogicalScalar(prhs[5]))
            mexErrMsgIdAndTxt("load_bl_tif:Transpose", "transposeFlag must be a logical scalar.");
        transpose = mxIsLogicalScalarTrue(prhs[5]);
    }

    const std::size_t numSlices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> fileList(numSlices);
    fileList.reserve(numSlices);
    for (std::size_t i = 0; i < numSlices; ++i)
    {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell))
            mexErrMsgIdAndTxt("load_bl_tif:Input", "File list must contain only strings.");
        MatlabString mstr(cell);
        if (!mstr.get() || !*mstr.get())
            mexErrMsgIdAndTxt("load_bl_tif:Input", "Filename in cell %zu is empty", i);
        fileList[i] = mstr.get();
    }

    int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    int roiW  = static_cast<int>(mxGetScalar(prhs[4]));

    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI parameters invalid");

    // Probe first slice for data type
    TIFFSetWarningHandler(nullptr);
    TiffHandle tif0(TIFFOpen(fileList[0].c_str(), "r"));
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open file %s (slice 0)", fileList[0].c_str());
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    getImageSize(tif0.get(), imgWidth, imgHeight);
    TIFFGetField(tif0.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif0.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 || (bitsPerSample != 8 && bitsPerSample != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
                          "Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported.");

    const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
    // Assume 1 sample per pixel: bitsPerSample is 8 or 16, so bytesPerPixel is 1 or 2
    const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;

    mwSize outH = transpose ? roiW : roiH;
    mwSize outW = transpose ? roiH : roiW;
    mwSize dims[3] = { outH, outW, static_cast<mwSize>(numSlices) };
    plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);

    void* outData = mxGetData(plhs[0]);
    std::size_t pixelsPerSlice = static_cast<std::size_t>(outH) * outW;

    std::fill_n(static_cast<uint8_t*>(outData), pixelsPerSlice * numSlices * bytesPerPixel, 0);

    // --- Slices loop with edge cropping and buffer offsets ---
    for (std::size_t z = 0; z < numSlices; ++z)
    {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:OpenFail", "Cannot open file %s (slice %zu)", fileList[z].c_str(), z);

        getImageSize(tif.get(), imgWidth, imgHeight);

        // --- Calculate overlap (intersection) between ROI and image ---
        std::size_t img_y_start = std::max(roiY0, 0);
        std::size_t img_y_end   = std::min(roiY0 + roiH - 1, static_cast<int>(imgHeight) - 1);
        std::size_t img_x_start = std::max(roiX0, 0);
        std::size_t img_x_end   = std::min(roiX0 + roiW - 1, static_cast<int>(imgWidth) - 1);

        std::size_t cropHz = img_y_end - img_y_start + 1; // How many rows to copy
        std::size_t cropWz = img_x_end - img_x_start + 1; // How many cols to copy

        if (cropHz <= 0 || cropWz <= 0) {
            mexErrMsgIdAndTxt("load_bl_tif:EmptyCrop", "Slice %zu has no overlap with ROI (%d,%d,%d,%d)",
                                z, roiX0+1, roiY0+1, roiW, roiH);
        }

        std::size_t out_row0 = std::max(0, img_y_start - roiY0);
        std::size_t out_col0 = std::max(0, img_x_start - roiX0);
        if (out_row0 + cropHz > roiH || out_col0 + cropWz > roiW)
            mexErrMsgIdAndTxt(
                "load_bl_tif:BoundsError",
                "Crop region (size %dx%d at offset %d,%d) exceeds output bounds (size %dx%d).",
                cropHz, cropWz, out_row0, out_col0, roiH, roiW);
        LoadTask task {
            img_y_start, img_x_start,
            out_row0, out_col0,
            cropHz, cropWz,
            roiH, roiW,
            z,
            outData,
            pixelsPerSlice,
            fileList[z],
            transpose
        };

        copySubRegion(task, tif.get(), bytesPerPixel);
    }
}
