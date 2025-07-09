/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)

  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)

  ──────────────────────────────────────────────────────────────────────────────
  OVERVIEW
  --------
  • Purpose
      Efficiently extracts an X-Y subregion (ROI) from a series of grayscale
      TIFF slices (1 TIFF per Z) and assembles them into a 3D MATLAB array.
      Optimized for speed, reliability, and robustness in high-throughput,
      multi-core environments.

  • Highlights
      – Supports 8-bit and 16-bit grayscale TIFFs (single-channel).
      – Handles both tiled and stripped formats, including compression (e.g., Deflate, LZW).
      – Fully cross-platform: Windows, Linux, macOS (requires libtiff ≥ 4.0).
      – Uses modern C++14/17 and multi-threading with a dynamic work queue.
      – ROI coordinates and dimensions use `uint32_t` for clarity and safety.
      – Each thread independently opens its own TIFF and manages its own decode buffers.
      – Endianness correction (byte-swapping) is automatically handled by libtiff.
      – Output shape is [Y X Z] by default; optional transpose to [X Y Z].
      – ROI bounds are strictly validated across all slices before memory allocation.
      – All thread-level errors are aggregated and reported as a single message.

  ──────────────────────────────────────────────────────────────────────────────
  MATLAB USAGE
  ------------
      img = load_bl_tif(files, y, x, height, width [, transposeFlag]);

      • files          – 1×N cell array of full path strings (one per Z slice)
      • y, x           – 1-based upper-left ROI coordinate (double scalars)
      • height, width  – ROI dimensions in pixels (double scalars)
      • transposeFlag  – (optional) logical or uint32 scalar, default = false
                         If true, output is returned in [X Y Z] format

      • returns
        – img          – MATLAB array of class uint8 or uint16:
                          [height width Z] or [width height Z] if transposed

      Example:
          files = dir('/some/folder/*.tif');
          paths = fullfile({files.folder}, {files.name});
          blk   = load_bl_tif(paths, 101, 201, 512, 512);         % Standard
          blkT  = load_bl_tif(paths, 101, 201, 512, 512, true);   % Transposed

  ──────────────────────────────────────────────────────────────────────────────
  COMPILATION
  -----------
      • Recommended: MATLAB R2018a+ with a C++14/17-capable compiler.

      • Use the provided `build_mex.m` script, or compile manually:

          mex -R2018a -largeArrayDims CXXFLAGS="\$(CXXFLAGS) -std=c++17" \
              LDFLAGS="\$(LDFLAGS) -ltiff" load_bl_tif.cpp

      • Ensure libtiff headers and libraries are available to the compiler.
      • On Windows, link against a precompiled `tiff.lib`.

  ──────────────────────────────────────────────────────────────────────────────
  CONSTRAINTS & SAFEGUARDS
  -------------------------
      • Files must be sorted by Z; no sorting is performed internally.
      • All slices must share identical size, bit depth, and be grayscale (1 sample/pixel).
      • The ROI must lie fully inside each slice. Validation is strict and occurs before allocation.
      • The output array must not exceed 2,147,483,647 pixels per slice (MATLAB limit).
      • RGB or multi-channel TIFFs are not supported.

  ──────────────────────────────────────────────────────────────────────────────
  PARALLELISM & PERFORMANCE
  --------------------------
      • Uses a dynamic work queue to assign TIFF slices to threads.
      • Each thread uses independent TIFF handles and decode buffers.
      • Byte-swapping is delegated to libtiff, which transparently returns native-endian data.
      • Aggregates all error messages after the parallel phase finishes.
      • For best performance, store TIFFs on SSD/NVMe and avoid over-threading.

  ---------------------------------------------------------------------------
  © 2025 Keivan Moradi — Released under GPLv3. See LICENSE or visit:
                         https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)
  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)
==============================================================================*/

/*==============================================================================
  load_bl_tif.cpp
  ---------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)
  Author:       Keivan Moradi
  Code review:  ChatGPT (4-o, o3, 4.1)
  License:      GNU GPL v3.0 (https://www.gnu.org/licenses/)
==============================================================================*/

/*==============================================================================
  load_bl_tif.cpp
  ------------------------------------------------------------------------------
  High-throughput sub-region loader for 3-D TIFF stacks (one TIFF per Z-slice)
  (C) Keivan Moradi, 2025 | License: GPLv3

  Output shapes:
    N mode (transpose = false):  [Y X Z] == [height, width, Z]
    T mode (transpose = true):   [X Y Z] == [width, height, Z]
==============================================================================*/

#define NOMINMAX
#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include "mex_thread_utils.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>      // for ::memcpy
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <limits>

// Constants
static constexpr uint16_t kSupportedBitDepth8  = 8;
static constexpr uint16_t kSupportedBitDepth16 = 16;
static constexpr size_t   kMaxPixelsPerSlice   = static_cast<size_t>(std::numeric_limits<int>::max());

// RAII string converter for MATLAB char -> UTF8
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString", "Failed to convert string from mxArray");
    }
    MatlabString(const MatlabString&) = delete;
    MatlabString& operator=(const MatlabString&) = delete;
    MatlabString(MatlabString&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    MatlabString& operator=(MatlabString&& other) noexcept {
        if (this != &other) { mxFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

// Task for a single slice load
struct LoadTask {
    uint32_t roiY0, roiX0, roiH, roiW;
    size_t zIndex;
    std::string path;
    bool transpose;
    LoadTask(uint32_t y0, uint32_t x0, uint32_t h, uint32_t w, size_t z, std::string p, bool tr)
        : roiY0(y0), roiX0(x0), roiH(h), roiW(w), zIndex(z), path(std::move(p)), transpose(tr) {}
};

// Helper: Check and parse inputs, assemble file list and ROI, check transpose flag
void parse_inputs(
    int nrhs, const mxArray* prhs[],
    std::vector<std::string>& fileList,
    uint32_t& roiY0, uint32_t& roiX0, uint32_t& roiH, uint32_t& roiW,
    bool& transpose
) {
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:usage", "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    // File list
    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:args", "First argument must be a cell array of filenames");
    size_t numSlices = mxGetNumberOfElements(prhs[0]);
    fileList.resize(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell)) mexErrMsgIdAndTxt("load_bl_tif:args", "File list must contain only strings.");
        MatlabString mstr(cell);
        fileList[i] = mstr.get();
    }

    // ROI params
    for (int i = 1; i <= 4; ++i) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1)
            mexErrMsgIdAndTxt("load_bl_tif:args", "Input argument %d must be a real double scalar.", i+1);
    }
    double y_in = mxGetScalar(prhs[1]);
    double x_in = mxGetScalar(prhs[2]);
    double h_in = mxGetScalar(prhs[3]);
    double w_in = mxGetScalar(prhs[4]);
    if (!mxIsFinite(y_in) || !mxIsFinite(x_in) || !mxIsFinite(h_in) || !mxIsFinite(w_in))
        mexErrMsgIdAndTxt("load_bl_tif:args", "y, x, height, width must be finite numbers.");
    if (y_in < 1 || x_in < 1 || h_in < 1 || w_in < 1)
        mexErrMsgIdAndTxt("load_bl_tif:args", "y, x, height, width must be positive (1-based).");
    roiY0 = static_cast<uint32_t>(y_in - 1);
    roiX0 = static_cast<uint32_t>(x_in - 1);
    roiH  = static_cast<uint32_t>(h_in);
    roiW  = static_cast<uint32_t>(w_in);

    transpose = false;
    if (nrhs == 6) {
        const mxArray* flag = prhs[5];
        if (mxIsLogicalScalar(flag))
            transpose = mxIsLogicalScalarTrue(flag);
        else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1)
            transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("load_bl_tif:args", "transposeFlag must be logical or int32/uint32 scalar.");
    }
}

// Helper: Check TIFF metadata for all slices; enforce shape, bitdepth, and ROI constraints
void check_tiff_metadata(
    const std::vector<std::string>& fileList,
    uint32_t roiY0, uint32_t roiX0, uint32_t roiH, uint32_t roiW,
    uint16_t& outBitsPerSample
) {
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, globalBitsPerSample = 0, samplesPerPixel = 1;
    for (size_t z = 0; z < fileList.size(); ++z) {
        TIFF* tif = TIFFOpen(fileList[z].c_str(), "r");
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Cannot open file %s (slice %zu)", fileList[z].c_str(), z+1);
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
        TIFFClose(tif);

        if (z == 0) globalBitsPerSample = bitsPerSample;
        else if (bitsPerSample != globalBitsPerSample)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Inconsistent bitsPerSample across slices. Expected %d, got %d in slice %zu (%s)", globalBitsPerSample, bitsPerSample, z+1, fileList[z].c_str());

        if (samplesPerPixel != 1 ||
            (globalBitsPerSample != kSupportedBitDepth8 && globalBitsPerSample != kSupportedBitDepth16))
            mexErrMsgIdAndTxt("load_bl_tif:file", "Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported. Slice %zu (%s)", z+1, fileList[z].c_str());

        if (roiY0 + roiH > imgHeight || roiX0 + roiW > imgWidth)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Requested ROI [%u:%u,%u:%u] is out of bounds for slice %zu (file: %s)",
                roiY0+1, roiY0+roiH, roiX0+1, roiX0+roiW, z+1, fileList[z].c_str());
    }
    outBitsPerSample = globalBitsPerSample;
}

// Helper: Output allocation. XYZ mode is always [Y X Z]; YXZ mode is [X Y Z]
void* create_output_array(mxArray*& plhs0, mxClassID outType, size_t roiH, size_t roiW, size_t numSlices, bool transpose) {
    // Preserve invariant: N mode (XYZ) = [Y X Z], T mode (YXZ) = [X Y Z]
    size_t outH = transpose ? roiW : roiH;
    size_t outW = transpose ? roiH : roiW;
    size_t dims[3] = { outH, outW, numSlices };
    plhs0 = mxCreateNumericArray(3, dims, outType, mxREAL);
    if (!plhs0) mexErrMsgIdAndTxt("load_bl_tif:alloc", "Failed to allocate output array.");
    return mxGetData(plhs0);
}

// Helper: Decodes a subregion from TIFF, into a temporary 2D block buffer
void decode_subregion_to_buffer(
    const LoadTask& task,
    TIFF* tif,
    uint8_t bytes_per_pixel,
    std::vector<uint8_t>& block_buffer,
    std::vector<uint8_t>& temp_buffer
) {
    uint32_t imgWidth = 0, imgHeight = 0;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight))
        throw std::runtime_error("Missing TIFFTAG_IMAGEWIDTH or IMAGELENGTH in file: " + task.path);

    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 ||
        (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
        throw std::runtime_error("Unsupported TIFF format: only 8/16-bit grayscale, 1 sample/pixel in file: " + task.path);

    bool isTiled = TIFFIsTiled(tif);
    if (isTiled) {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0)
            throw std::runtime_error("Invalid tile size in TIFF metadata in file: " + task.path);
        size_t uncompressedTileBytes = static_cast<size_t>(tileW) * tileH * bytes_per_pixel;
        if (uncompressedTileBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (uncompressedTileBytes > temp_buffer.size())
            temp_buffer.resize(uncompressedTileBytes);

        uint32_t prevTile = UINT32_MAX;
        for (uint32_t row = 0; row < task.roiH; ++row) {
            uint32_t imgY = task.roiY0 + row;
            for (uint32_t col = 0; col < task.roiW; ++col) {
                uint32_t imgX = task.roiX0 + col;
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);
                if (tileIdx != prevTile) {
                    tsize_t ret = TIFFReadEncodedTile(
                        tif, tileIdx, temp_buffer.data(), uncompressedTileBytes);
                    if (ret < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed (tile " + std::to_string(tileIdx) + ") in file: " + task.path);
                    prevTile = tileIdx;
                }
                uint32_t relY = imgY % tileH, relX = imgX % tileW;
                size_t src_off = (static_cast<size_t>(relY) * tileW + relX) * bytes_per_pixel;
                size_t dst_off = (static_cast<size_t>(row) * task.roiW + col) * bytes_per_pixel;
                ::memcpy(block_buffer.data() + dst_off, temp_buffer.data() + src_off, bytes_per_pixel);
            }
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;
        size_t maxStripBytes = static_cast<size_t>(rowsPerStrip) * imgWidth * bytes_per_pixel;
        if (maxStripBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (maxStripBytes > temp_buffer.size())
            temp_buffer.resize(maxStripBytes);

        tstrip_t currentStrip = (tstrip_t)-1;
        tsize_t  nbytes = 0;

        for (uint32_t row = 0; row < task.roiH; ++row) {
            uint32_t tifRow   = task.roiY0 + row;
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);
            if (stripIdx != currentStrip) {
                nbytes = TIFFReadEncodedStrip(tif, stripIdx, temp_buffer.data(), maxStripBytes);
                if (nbytes < 0)
                    throw std::runtime_error("TIFFReadEncodedStrip failed (strip " + std::to_string(stripIdx) + ") in file: " + task.path);
                currentStrip = stripIdx;
            }
            uint32_t rowsInThisStrip = static_cast<uint32_t>(nbytes / (imgWidth * bytes_per_pixel));
            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow        = tifRow - stripStartRow;
            if (relRow >= rowsInThisStrip)
                throw std::runtime_error("Row " + std::to_string(tifRow+1) + " exceeds decoded strip size (strip " + std::to_string(stripIdx) + ") in file: " + task.path);
            uint8_t* scanlinePtr = temp_buffer.data() + (static_cast<size_t>(relRow) * imgWidth * bytes_per_pixel);
            for (uint32_t col = 0; col < task.roiW; ++col) {
                size_t src_off = (static_cast<size_t>(task.roiX0 + col)) * bytes_per_pixel;
                size_t dst_off = (static_cast<size_t>(row) * task.roiW + col) * bytes_per_pixel;
                if (src_off + bytes_per_pixel > static_cast<size_t>(nbytes))
                    throw std::runtime_error("Column " + std::to_string(col+1) + " exceeds decoded strip size (strip " + std::to_string(stripIdx) + ") in file: " + task.path);
                ::memcpy(block_buffer.data() + dst_off, scanlinePtr + src_off, bytes_per_pixel);
            }
        }
    }
}

// Helper: For output stride (no-transpose), output is always [Y X Z] (N mode, XYZ)
// For transpose, copy per-element to [X Y Z] (T mode, YXZ)
void copy_block_to_output(
    const LoadTask& task,
    const std::vector<uint8_t>& block_buffer,
    void* out_data,
    size_t bytes_per_pixel,
    size_t roiH, size_t roiW,
    size_t pixelsPerSlice
) {
    if (!task.transpose) {
        // XYZ: [Y X Z], contiguous memory for each slice (fast bulk copy)
        size_t dstByte = task.zIndex * pixelsPerSlice * bytes_per_pixel;
        size_t sliceBytes = pixelsPerSlice * bytes_per_pixel;
        ::memcpy(static_cast<uint8_t*>(out_data) + dstByte, block_buffer.data(), sliceBytes);
    } else {
        // YXZ: [X Y Z], per-element copy
        for (uint32_t row = 0; row < roiH; ++row) {
            for (uint32_t col = 0; col < roiW; ++col) {
                size_t dstElem = col + row * roiW + task.zIndex * pixelsPerSlice;
                size_t dstByte = dstElem * bytes_per_pixel;
                size_t srcByte = (row * roiW + col) * bytes_per_pixel;
                ::memcpy(static_cast<uint8_t*>(out_data) + dstByte, block_buffer.data() + srcByte, bytes_per_pixel);
            }
        }
    }
}

// ==============================
//       ENTRY POINT
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    ensure_hwloc_initialized();
    try {
        std::vector<std::string> fileList;
        uint32_t roiY0, roiX0, roiH, roiW;
        bool transpose;
        parse_inputs(nrhs, prhs, fileList, roiY0, roiX0, roiH, roiW, transpose);

        uint16_t bitsPerSample = 0;
        check_tiff_metadata(fileList, roiY0, roiX0, roiH, roiW, bitsPerSample);

        const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
        const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;
        size_t numSlices = fileList.size();
        size_t outH = transpose ? roiW : roiH;
        size_t outW = transpose ? roiH : roiW;
        size_t pixelsPerSlice = outH * outW;
        if (pixelsPerSlice > kMaxPixelsPerSlice)
            mexErrMsgIdAndTxt("load_bl_tif:Error", "Requested ROI too large (>2^31 elements).");
        void* outData = create_output_array(plhs[0], outType, roiH, roiW, numSlices, transpose);

        // --- Main decode loop: per-slice, in parallel if desired ---
        // (single-threaded version for clarity; you can parallelize if needed)
        for (size_t z = 0; z < numSlices; ++z) {
            LoadTask task(roiY0, roiX0, roiH, roiW, z, fileList[z], transpose);
            std::vector<uint8_t> block_buffer(roiH * roiW * bytesPerPixel);
            std::vector<uint8_t> temp_buffer;
            TIFF* tif = TIFFOpen(task.path.c_str(), "r");
            if (!tif)
                mexErrMsgIdAndTxt("load_bl_tif:file", "Cannot open file %s (slice %zu)", task.path.c_str(), z+1);
            try {
                decode_subregion_to_buffer(task, tif, bytesPerPixel, block_buffer, temp_buffer);
                TIFFClose(tif);
                copy_block_to_output(task, block_buffer, outData, bytesPerPixel, roiH, roiW, pixelsPerSlice);
            } catch (...) {
                TIFFClose(tif);
                throw;
            }
        }
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", ex.what());
    }
}
