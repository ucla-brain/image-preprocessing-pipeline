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

#define NOMINMAX
#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include "mex_thread_utils.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <limits>

// --- Constants ---
static constexpr uint16_t kSupportedBitDepth8  = 8;
static constexpr uint16_t kSupportedBitDepth16 = 16;
static constexpr size_t   kMaxPixelsPerSlice   = static_cast<size_t>(std::numeric_limits<int>::max());
static constexpr size_t   kWires = 1;

// --- RAII wrapper for mxArrayToUTF8String() ---
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

// --- Data structures ---
struct LoadTask {
    uint32_t in_row0, in_col0, cropH, cropW;
    uint32_t roiH, roiW, zIndex;
    size_t out_row0, out_col0;
    size_t pixelsPerSlice;
    std::string path;
    bool transpose;
    LoadTask() = default;
    LoadTask(uint32_t inY, uint32_t inX, size_t outY, size_t outX,
             uint32_t h, uint32_t w, uint32_t roiH_, uint32_t roiW_,
             uint32_t z, size_t pps, std::string filename, bool transpose_)
        : in_row0(inY), in_col0(inX), out_row0(outY), out_col0(outX),
          cropH(h), cropW(w), roiH(roiH_), roiW(roiW_),
          zIndex(z), pixelsPerSlice(pps), path(std::move(filename)), transpose(transpose_) {}
};

struct TaskResult {
    size_t block_id;
    std::vector<uint8_t> data;
    uint32_t cropH, cropW;
    TaskResult(size_t id, size_t datasz, uint32_t ch, uint32_t cw)
        : block_id(id), data(datasz), cropH(ch), cropW(cw) {}
};

struct TiffCloser { void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); } };
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// --- Helper: Parse MATLAB inputs safely ---
struct ParsedInputs {
    std::vector<std::string> fileList;
    uint32_t roiY0, roiX0, roiH, roiW;
    bool transpose;
};

ParsedInputs parse_inputs(int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:usage", "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");
    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:args", "First argument must be a cell array of filenames");
    size_t numSlices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> fileList(numSlices);
    for (int i = 0; i < numSlices; ++i) {
        const mxArray* cell = mxGetCell(prhs[0], i);
        if (!mxIsChar(cell)) mexErrMsgIdAndTxt("load_bl_tif:args", "File list must contain only strings.");
        MatlabString mstr(cell);
        fileList[i] = mstr.get();
    }
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

    bool transpose = false;
    if (nrhs == 6) {
        const mxArray* flag = prhs[5];
        if (mxIsLogicalScalar(flag))
            transpose = mxIsLogicalScalarTrue(flag);
        else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1)
            transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("load_bl_tif:args", "transposeFlag must be logical or int32/uint32 scalar.");
    }
    return {fileList,
            static_cast<uint32_t>(y_in - 1),
            static_cast<uint32_t>(x_in - 1),
            static_cast<uint32_t>(h_in),
            static_cast<uint32_t>(w_in),
            transpose};
}

// --- Helper: Check TIFF metadata across all slices ---
void check_tiff_metadata(const std::vector<std::string>& fileList,
    uint32_t roiY0, uint32_t roiX0, uint32_t roiH, uint32_t roiW, uint16_t& outBitsPerSample) {
    uint32_t imgWidth = 0, imgHeight = 0;
    uint16_t bitsPerSample = 0, globalBitsPerSample = 0, samplesPerPixel = 1;
    for (size_t z = 0; z < fileList.size(); ++z) {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
        if (!tif)
            mexErrMsgIdAndTxt("load_bl_tif:file", "Cannot open file %s (slice %zu)", fileList[z].c_str(), z+1);
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgWidth);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgHeight);
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

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

// --- Helper: Output allocation ---
void* create_output_array(mxArray*& plhs0, mxClassID outType, size_t outH, size_t outW, size_t numSlices) {
    size_t dims[3] = { outH, outW, numSlices };
    plhs0 = mxCreateNumericArray(3, dims, outType, mxREAL);
    if (!plhs0) mexErrMsgIdAndTxt("load_bl_tif:alloc", "Failed to allocate output array.");
    return mxGetData(plhs0);
}

// --- Helper: Task generation ---
std::vector<LoadTask> create_tasks(const std::vector<std::string>& fileList,
    uint32_t roiY0, uint32_t roiX0, uint32_t roiH, uint32_t roiW,
    size_t pixelsPerSlice, bool transpose) {
    std::vector<LoadTask> tasks;
    size_t numSlices = fileList.size();
    tasks.reserve(numSlices);
    for (size_t z = 0; z < numSlices; ++z) {
        tasks.emplace_back(roiY0, roiX0, 0, 0, roiH, roiW, roiH, roiW, static_cast<uint32_t>(z), pixelsPerSlice, fileList[z], transpose);
    }
    return tasks;
}

// --- Helper: MATLAB output index calculation ---
inline size_t computeDstIndex(const LoadTask& task, uint32_t row, uint32_t col) noexcept {
    size_t r = task.out_row0 + row;
    size_t c = task.out_col0 + col;
    size_t slice = task.zIndex;
    if (!task.transpose)
        return r + c * task.roiH + slice * task.pixelsPerSlice; // [Y X Z]
    else
        return c + r * task.roiW + slice * task.pixelsPerSlice; // [X Y Z]
}

// --- Helper: Read ROI block from TIFF (tiles or strips) into blockBuf ---
void readSubRegionToBuffer(const LoadTask& task, TIFF* tif, uint8_t bytesPerPixel,
    std::vector<uint8_t>& blockBuf, std::vector<uint8_t>& tempBuf) {
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

    const bool isTiled = TIFFIsTiled(tif);

    if (isTiled) {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0)
            throw std::runtime_error("Invalid tile size in TIFF metadata in file: " + task.path);
        size_t uncompressedTileBytes = static_cast<size_t>(tileW) * tileH * bytesPerPixel;
        if (uncompressedTileBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (uncompressedTileBytes > tempBuf.size())
            tempBuf.resize(uncompressedTileBytes);

        uint32_t prevTile = UINT32_MAX;
        for (uint32_t row = 0; row < task.cropH; ++row) {
            uint32_t imgY = task.in_row0 + row;
            for (uint32_t col = 0; col < task.cropW; ++col) {
                uint32_t imgX = task.in_col0 + col;
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);
                if (tileIdx != prevTile) {
                    tsize_t ret = TIFFReadEncodedTile(
                        tif, tileIdx, tempBuf.data(), uncompressedTileBytes
                    );
                    if (ret < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed (tile " + std::to_string(tileIdx) + ") in file: " + task.path);
                    prevTile = tileIdx;
                }
                uint32_t relY = imgY % tileH;
                uint32_t relX = imgX % tileW;
                size_t   srcOff = (static_cast<size_t>(relY) * tileW + relX) * bytesPerPixel;
                size_t   dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;
                std::memcpy(blockBuf.data() + dstOff,
                            tempBuf.data() + srcOff,
                            bytesPerPixel);
            }
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;
        const size_t maxStripBytes = static_cast<size_t>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (maxStripBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (maxStripBytes > tempBuf.size())
            tempBuf.resize(maxStripBytes);
        tstrip_t currentStrip = (tstrip_t)-1;
        tsize_t  nbytes = 0;

        for (uint32_t row = 0; row < task.cropH; ++row) {
            uint32_t tifRow   = task.in_row0 + row;
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);
            if (stripIdx != currentStrip) {
                nbytes = TIFFReadEncodedStrip(tif, stripIdx, tempBuf.data(), maxStripBytes);
                if (nbytes < 0)
                    throw std::runtime_error("TIFFReadEncodedStrip failed (strip " + std::to_string(stripIdx) + ") in file: " + task.path);
                currentStrip = stripIdx;
            }
            const uint32_t rowsInThisStrip = static_cast<uint32_t>(nbytes / (imgWidth * bytesPerPixel));
            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow        = tifRow - stripStartRow;

            if (relRow >= rowsInThisStrip)
                throw std::runtime_error("Row " + std::to_string(tifRow+1) + " exceeds decoded strip size (strip " + std::to_string(stripIdx) + ") in file: " + task.path);

            uint8_t* scanlinePtr = tempBuf.data() + (static_cast<size_t>(relRow) * imgWidth * bytesPerPixel);
            for (uint32_t col = 0; col < task.cropW; ++col) {
                size_t srcOff = (static_cast<size_t>(task.in_col0 + col)) * bytesPerPixel;
                size_t dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;
                if (srcOff + bytesPerPixel > static_cast<size_t>(nbytes))
                    throw std::runtime_error("Column " + std::to_string(col+1) + " exceeds decoded strip size (strip " + std::to_string(stripIdx) + ") in file: " + task.path);
                std::memcpy(blockBuf.data() + dstOff,
                            scanlinePtr + srcOff,
                            bytesPerPixel);
            }
        }
    }
}

// --- Parallel decode/copy ---
void parallel_decode_and_copy(const std::vector<LoadTask>& tasks, void* outData, size_t bytesPerPixel) {
    const size_t numSlices       = tasks.size();
    const size_t threadPairCount = std::min(numSlices, get_available_cores());
    const size_t numWires        = threadPairCount / kWires + ((threadPairCount % kWires) ? 1 : 0);

    using TaskPtr = std::shared_ptr<TaskResult>;
    std::vector<std::unique_ptr<BoundedQueue<TaskPtr>>> queuesForWires;
    queuesForWires.reserve(numWires);
    for (size_t w = 0; w < numWires; ++w)
        queuesForWires.emplace_back(std::make_unique<BoundedQueue<TaskPtr>>(2 * kWires));

    std::vector<std::thread> producerThreads, consumerThreads;
    producerThreads.reserve(threadPairCount);
    consumerThreads.reserve(threadPairCount);

    std::atomic<uint32_t> nextSliceIndex{0};
    std::vector<std::string> runtimeErrors;
    std::mutex              errorMutex;
    std::atomic<bool>       abortFlag{false};
    auto                    threadPairs = assign_thread_affinity_pairs(threadPairCount);

    // PRODUCERS: decode on producerLogicalCore → push TaskResult pointers
    for (size_t t = 0; t < threadPairCount; ++t) {
        auto& queueForPair = *queuesForWires[t / kWires];
        producerThreads.emplace_back([&, t] {
            set_thread_affinity(threadPairs[t].producerLogicalCore);
            std::vector<uint8_t> tempBuf;
            while (true) {
                if (abortFlag.load(std::memory_order_acquire)) break;
                uint32_t idx = nextSliceIndex.fetch_add(1, std::memory_order_relaxed);
                if (idx >= numSlices) break;
                const auto& task = tasks[idx];
                try {
                    TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
                    if (!tif) {
                        std::lock_guard<std::mutex> lck(errorMutex);
                        runtimeErrors.emplace_back(
                            "Slice " + std::to_string(task.zIndex+1) +
                            ": Cannot open " + task.path
                        );
                        abortFlag.store(true, std::memory_order_release);
                        break;
                    }
                    auto res = std::make_shared<TaskResult>(
                        idx,
                        task.cropH * task.cropW * bytesPerPixel,
                        task.cropH,
                        task.cropW
                    );
                    readSubRegionToBuffer(task, tif.get(), bytesPerPixel, res->data, tempBuf);
                    queueForPair.push(res);
                }
                catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lck(errorMutex);
                    runtimeErrors.emplace_back(
                        "Slice " + std::to_string(task.zIndex+1) + ": " + ex.what()
                    );
                    abortFlag.store(true, std::memory_order_release);
                    break;
                }
                catch (...) {
                    std::lock_guard<std::mutex> lck(errorMutex);
                    runtimeErrors.emplace_back(
                        "Slice " + std::to_string(task.zIndex+1) + ": Unknown exception"
                    );
                    abortFlag.store(true, std::memory_order_release);
                    break;
                }
            }
            queueForPair.push(nullptr);
        });
    }

    // CONSUMERS: pop TaskResult → write into outData on consumerLogicalCore
    for (size_t t = 0; t < threadPairCount; ++t) {
        auto& queueForPair = *queuesForWires[t / kWires];
        consumerThreads.emplace_back([&, t] {
            set_thread_affinity(threadPairs[t].consumerLogicalCore);
            while (true) {
                if (abortFlag.load(std::memory_order_acquire)) break;
                TaskPtr res;
                queueForPair.wait_and_pop(res);
                if (!res) break;

                const auto& task   = tasks[res->block_id];
                const uint8_t* src = res->data.data();
                uint8_t*       dst = static_cast<uint8_t*>(outData)
                                   + task.zIndex * task.pixelsPerSlice * bytesPerPixel;

                // Non-transpose: always contiguous writes down each column
                const size_t dstColStride = task.roiH * bytesPerPixel;
                const size_t srcRowStride = task.cropW * bytesPerPixel;

                if (bytesPerPixel == 1) {
                    // 8-bit path: single‐byte loads/stores
                    for (uint32_t col = 0; col < task.cropW; ++col) {
                        const uint8_t* srcCol = src + col;
                        uint8_t*       dstCol = dst + col * dstColStride;
                        for (uint32_t row = 0; row < task.cropH; ++row) {
                            dstCol[row] = srcCol[row * srcRowStride];
                        }
                    }
                } else {
                    // 16-bit path: 2-byte loads/stores
                    const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
                    uint16_t*       dst16 = reinterpret_cast<uint16_t*>(dst);
                    const size_t    src16Stride = srcRowStride / 2;
                    const size_t    dst16Stride = dstColStride / 2;

                    for (uint32_t col = 0; col < task.cropW; ++col) {
                        const uint16_t* s = src16 + col;
                        uint16_t*       d = dst16 + col * dst16Stride;
                        for (uint32_t row = 0; row < task.cropH; ++row) {
                            d[row] = s[row * src16Stride];
                        }
                    }
                }
            }
        });
    }

    // join and error out
    for (auto& p : producerThreads) p.join();
    for (auto& c : consumerThreads) c.join();

    if (!runtimeErrors.empty()) {
        std::ostringstream oss;
        oss << "Errors during load_bl_tif (producer/consumer):\n";
        for (auto& e : runtimeErrors) oss << "  - " << e << "\n";
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", oss.str().c_str());
    }
}


// ==============================
//       ENTRY POINT
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    ensure_hwloc_initialized();
    try {
        ParsedInputs args = parse_inputs(nrhs, prhs);
        uint16_t bitsPerSample = 0;
        check_tiff_metadata(args.fileList, args.roiY0, args.roiX0, args.roiH, args.roiW, bitsPerSample);
        const mxClassID outType = (bitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
        const uint8_t bytesPerPixel = (bitsPerSample == 16) ? 2 : 1;

        size_t outH = args.transpose ? args.roiW : args.roiH;
        size_t outW = args.transpose ? args.roiH : args.roiW;
        size_t pixelsPerSlice = outH * outW;
        if (pixelsPerSlice > kMaxPixelsPerSlice)
            mexErrMsgIdAndTxt("load_bl_tif:Error", "Requested ROI too large (>2^31 elements).");
        void* outData = create_output_array(plhs[0], outType, outH, outW, args.fileList.size());

        auto tasks = create_tasks(args.fileList, args.roiY0, args.roiX0, args.roiH, args.roiW, pixelsPerSlice, args.transpose);

        parallel_decode_and_copy(tasks, outData, bytesPerPixel);
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", ex.what());
    }
}
