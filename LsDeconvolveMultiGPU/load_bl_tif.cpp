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
#include <condition_variable>
#include <exception>
#include <atomic>
#include <sstream>

// --- Config ---
static constexpr uint16_t kSupportedBitDepth8  = 8;
static constexpr uint16_t kSupportedBitDepth16 = 16;
static constexpr size_t kMaxPixelsPerSlice = static_cast<size_t>(std::numeric_limits<int>::max());
static constexpr size_t kWires = 1;  // Set to 1 for maximal locality, or >1 for more queues per NUMA node

// RAII wrapper for mxArrayToUTF8String()
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString",
                              "Failed to convert string from mxArray");
    }
    // Prevent accidental copies (which would double-free)
    MatlabString(const MatlabString&)            = delete;
    MatlabString& operator=(const MatlabString&) = delete;
    // Allow moves
    MatlabString(MatlabString&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    MatlabString& operator=(MatlabString&& other) noexcept {
        if (this != &other) {
            mxFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

struct LoadTask {
    uint32_t in_row0, in_col0, cropH, cropW;   // PATCH: Now uint32_t
    uint32_t roiH, roiW, zIndex;
    size_t out_row0, out_col0;
    size_t pixelsPerSlice;
    std::string path;
    bool transpose;
    LoadTask() = default;
    LoadTask(
        uint32_t inY, uint32_t inX, size_t outY, size_t outX, // PATCH
        uint32_t h, uint32_t w, uint32_t roiH_, uint32_t roiW_,
        uint32_t z, size_t pps, std::string filename, bool transpose_
    ) : in_row0(inY), in_col0(inX), out_row0(outY), out_col0(outX),
        cropH(h), cropW(w), roiH(roiH_), roiW(roiW_),
        zIndex(z), pixelsPerSlice(pps), path(std::move(filename)), transpose(transpose_) {}
};

struct TiffCloser {
    void operator()(TIFF* tif) const { if (tif) TIFFClose(tif); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// ------------------------------------------------------------------
//  64-bit-safe destination index utility for output array indexing (portable for both shapes)
// ------------------------------------------------------------------
inline size_t computeDstIndex(const LoadTask& task,
                              uint32_t row, uint32_t col) noexcept  // PATCH
{
    size_t r = task.out_row0 + row;
    size_t c = task.out_col0 + col;
    size_t slice = task.zIndex;

    // MATLAB arrays are column-major and transpose swaps [Y, X] ↔ [X, Y]
    if (!task.transpose)
        return r + c * task.roiH + slice * task.pixelsPerSlice;
    else
        return c + r * task.roiW + slice * task.pixelsPerSlice;
}

// The result buffer for each block
struct TaskResult {
    size_t block_id; // index into task/result vector
    std::vector<uint8_t> data;
    uint32_t cropH, cropW; // PATCH: uint32_t for consistency with task
    TaskResult(size_t id, size_t datasz, uint32_t ch, uint32_t cw)
        : block_id(id), data(datasz), cropH(ch), cropW(cw) {}
};

// ---------------------------------------------------------------------------
//  Safe sub-region reader for both tiled and stripped TIFFs
// ---------------------------------------------------------------------------
static void readSubRegionToBuffer(
    const LoadTask& task,
    TIFF* tif,
    uint8_t bytesPerPixel,
    std::vector<uint8_t>& blockBuf,
    std::vector<uint8_t>& tempBuf
)
{
    // PATCH: Use uint32_t for TIFF dimension variables
    uint32_t imgWidth = 0, imgHeight = 0;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgWidth) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight))
    {
        throw std::runtime_error("Missing TIFFTAG_IMAGEWIDTH or IMAGELENGTH in file: " + task.path);
    }

    uint16_t bitsPerSample = 0, samplesPerPixel = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (samplesPerPixel != 1 ||
        (bitsPerSample != kSupportedBitDepth8 && bitsPerSample != kSupportedBitDepth16))
    {
        throw std::runtime_error("Unsupported TIFF format: only 8/16-bit grayscale, 1 sample/pixel in file: " + task.path);
    }

    const bool isTiled = TIFFIsTiled(tif);

    if (isTiled)
    {
        uint32_t tileW = 0, tileH = 0; // PATCH
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0)
            throw std::runtime_error("Invalid tile size in TIFF metadata in file: " + task.path);

        size_t uncompressedTileBytes = static_cast<size_t>(tileW) * tileH * bytesPerPixel;
        if (uncompressedTileBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (uncompressedTileBytes > tempBuf.size())
            tempBuf.resize(uncompressedTileBytes);
        const size_t nTilePixels = uncompressedTileBytes / bytesPerPixel;

        uint32_t prevTile = UINT32_MAX;

        for (uint32_t row = 0; row < task.cropH; ++row) {  // PATCH
            uint32_t imgY = task.in_row0 + row;
            for (uint32_t col = 0; col < task.cropW; ++col) {
                uint32_t imgX = task.in_col0 + col;
                uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (tileIdx != prevTile) {
                    tsize_t ret = TIFFReadEncodedTile(
                        tif,
                        tileIdx,
                        tempBuf.data(),
                        uncompressedTileBytes
                    );
                    if (ret < 0)
                    {
                        std::ostringstream oss;
                        oss << "TIFFReadEncodedTile failed (tile " << tileIdx << ") in file: " << task.path;
                        throw std::runtime_error(oss.str());
                    }
                    size_t validBytes = static_cast<size_t>(ret);
                    size_t validPixels = validBytes / bytesPerPixel;
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
    }
    else
    {
        uint32_t rowsPerStrip = 0; // PATCH
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = imgHeight;

        const size_t maxStripBytes = static_cast<size_t>(rowsPerStrip) * imgWidth * bytesPerPixel;
        if (maxStripBytes > static_cast<size_t>(std::numeric_limits<tsize_t>::max()))
            throw std::runtime_error("Tile buffer too large (overflow risk)");
        if (maxStripBytes > tempBuf.size())
            tempBuf.resize(maxStripBytes);
        tstrip_t currentStrip = (tstrip_t)-1;
        tsize_t  nbytes = 0;

        for (uint32_t row = 0; row < task.cropH; ++row) // PATCH
        {
            uint32_t tifRow   = task.in_row0 + row;
            tstrip_t stripIdx = TIFFComputeStrip(tif, tifRow, 0);

            if (stripIdx != currentStrip) {
                nbytes = TIFFReadEncodedStrip(tif, stripIdx, tempBuf.data(), maxStripBytes);
                if (nbytes < 0) {
                    std::ostringstream oss;
                    oss << "TIFFReadEncodedStrip failed (strip " << stripIdx << ") in file: " << task.path;
                    throw std::runtime_error(oss.str());
                }
                currentStrip = stripIdx;
            }

            const uint32_t rowsInThisStrip =
                static_cast<uint32_t>(nbytes / (imgWidth * bytesPerPixel));
            uint32_t stripStartRow = stripIdx * rowsPerStrip;
            uint32_t relRow        = tifRow - stripStartRow;

            if (relRow >= rowsInThisStrip) {
                std::ostringstream oss;
                oss << "Row " << tifRow+1 << " exceeds decoded strip size (strip " << stripIdx << ") in file: " << task.path;
                throw std::runtime_error(oss.str());
            }

            uint8_t* scanlinePtr = tempBuf.data() +
                (static_cast<size_t>(relRow) * imgWidth * bytesPerPixel);

            for (uint32_t col = 0; col < task.cropW; ++col) // PATCH
            {
                size_t srcOff = (static_cast<size_t>(task.in_col0 + col)) * bytesPerPixel;
                size_t dstOff = (static_cast<size_t>(row) * task.cropW + col) * bytesPerPixel;

                if (srcOff + bytesPerPixel > static_cast<size_t>(nbytes)) {
                    std::ostringstream oss;
                    oss << "Column " << col+1 << " exceeds decoded strip size (strip " << stripIdx << ") in file: " << task.path;
                    throw std::runtime_error(oss.str());
                }

                std::memcpy(blockBuf.data() + dstOff,
                            scanlinePtr + srcOff,
                            bytesPerPixel);
            }
        }
    }
}

void worker_main(
    const std::vector<LoadTask>& tasks,
    std::vector<TaskResult>& results,
    uint8_t bytesPerPixel,
    std::mutex& err_mutex,
    std::vector<std::string>& errors,
    std::atomic<size_t>& error_count,
    std::atomic<size_t>& nextTask
) {
    std::vector<uint8_t> tempBuf;

    size_t i;
    while ((i = nextTask.fetch_add(1)) < tasks.size()) {
        const auto& task = tasks[i];
        try {
            TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
            if (!tif) {
                std::lock_guard<std::mutex> lck(err_mutex);
                errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) +
                                    ": Cannot open file " + task.path);
                error_count++;
                continue;
            }

            readSubRegionToBuffer(task, tif.get(), bytesPerPixel, results[i].data, tempBuf);

        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": " + ex.what());
            error_count++;
        } catch (...) {
            std::lock_guard<std::mutex> lck(err_mutex);
            errors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": Unknown exception");
            error_count++;
        }
    }
}


// ==============================
//       ENTRY POINT
// ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ensure_hwloc_initialized();
    try {
        if (nrhs < 5 || nrhs > 6)
            throw std::runtime_error("Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

        if (!mxIsCell(prhs[0]))
            throw std::runtime_error("First argument must be a cell array of filenames");

        bool transpose = false;
        if (nrhs == 6) {
            const mxArray* flag = prhs[5];

            if (mxIsLogicalScalar(flag)) {
                transpose = mxIsLogicalScalarTrue(flag);
            } else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1) {
                transpose = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
            } else {
                throw std::runtime_error("transposeFlag must be a logical or int32/uint32 scalar.");
            }
        }

        size_t numSlices = static_cast<size_t>(mxGetNumberOfElements(prhs[0]));
        std::vector<std::string> fileList(numSlices);
        for (int i = 0; i < numSlices; ++i) {
            const mxArray* cell = mxGetCell(prhs[0], i);
            if (!mxIsChar(cell))
                throw std::runtime_error("File list must contain only strings.");
            MatlabString mstr(cell);
            if (!mstr.get() || !*mstr.get()) {
                throw std::runtime_error("Filename in cell " + std::to_string(i + 1) + " is empty");
            }
            fileList[i] = mstr.get();
        }

        for (int i = 1; i <= 4; ++i) {
            if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) || mxGetNumberOfElements(prhs[i]) != 1)
                throw std::runtime_error("Input argument " + std::to_string(i + 1) + " must be a real double scalar.");
        }

        double y_in = mxGetScalar(prhs[1]);
        double x_in = mxGetScalar(prhs[2]);
        double h_in = mxGetScalar(prhs[3]);
        double w_in = mxGetScalar(prhs[4]);

        if (!mxIsFinite(y_in) || !mxIsFinite(x_in) || !mxIsFinite(h_in) || !mxIsFinite(w_in))
            throw std::runtime_error("y, x, height, width must be finite numbers.");

        if (y_in < 1 || x_in < 1 || h_in < 1 || w_in < 1)
            throw std::runtime_error("y, x, height, width must be positive (1-based).");

        uint32_t roiY0 = static_cast<uint32_t>(y_in - 1);
        uint32_t roiX0 = static_cast<uint32_t>(x_in - 1);
        uint32_t roiH  = static_cast<uint32_t>(h_in);
        uint32_t roiW  = static_cast<uint32_t>(w_in);

        uint32_t imgWidth = 0, imgHeight = 0;
        uint16_t bitsPerSample = 0, globalBitsPerSample = 0, samplesPerPixel = 1;
        for (size_t z = 0; z < numSlices; ++z) {
            TiffHandle tif(TIFFOpen(fileList[z].c_str(), "r"));
            if (!tif) {
                throw std::runtime_error("Cannot open file " + fileList[z] + " (slice " + std::to_string(z + 1) + ")");
            }
            TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgWidth);
            TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgHeight);
            TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
            TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

            if (z == 0) {
                globalBitsPerSample = bitsPerSample;
            } else if (bitsPerSample != globalBitsPerSample) {
                throw std::runtime_error("Inconsistent bitsPerSample across slices. Expected " +
                    std::to_string(globalBitsPerSample) + ", got " + std::to_string(bitsPerSample) +
                    " in slice " + std::to_string(z + 1) + " (" + fileList[z] + ")");
            }

            if (samplesPerPixel != 1 ||
                (globalBitsPerSample != kSupportedBitDepth8 && globalBitsPerSample != kSupportedBitDepth16)) {
                throw std::runtime_error("Only 8/16-bit grayscale TIFFs (1 sample per pixel) are supported. Slice " +
                    std::to_string(z + 1) + " (" + fileList[z] + ")");
            }

            if (roiY0 + roiH > imgHeight || roiX0 + roiW > imgWidth) {
                throw std::runtime_error("Requested ROI [" +
                    std::to_string(roiY0 + 1) + ":" + std::to_string(roiY0 + roiH) + "," +
                    std::to_string(roiX0 + 1) + ":" + std::to_string(roiX0 + roiW) +
                    "] is out of bounds for slice " + std::to_string(z + 1) + " (file: " + fileList[z] + ")");
            }
        }

        const mxClassID outType = (globalBitsPerSample == 8) ? mxUINT8_CLASS : mxUINT16_CLASS;
        const uint8_t bytesPerPixel = (globalBitsPerSample == 16) ? 2 : 1;

        size_t outH = transpose ? roiW : roiH;
        size_t outW = transpose ? roiH : roiW;
        size_t dims[3] = { outH, outW, numSlices };
        plhs[0] = mxCreateNumericArray(3, dims, outType, mxREAL);
        if (!plhs[0])
            throw std::runtime_error("Failed to allocate output array.");

        void* outData = mxGetData(plhs[0]);
        size_t pixelsPerSlice = outH * outW;
        if (pixelsPerSlice > kMaxPixelsPerSlice)
            throw std::runtime_error("Requested ROI too large (>2^31 elements).");

        // ---- Prepare all tasks ----
        std::vector<LoadTask> tasks;
        tasks.reserve(numSlices);
        for (size_t z = 0; z < numSlices; ++z) {
            tasks.emplace_back(roiY0, roiX0, 0, 0, roiH, roiW, roiH, roiW,
                               static_cast<uint32_t>(z), pixelsPerSlice, fileList[z], transpose);
        }

        // ==============================
        //       PRODUCER-CONSUMER DISPATCH
        // ==============================
        const size_t threadPairCount = std::min(numSlices, get_available_cores());
        const size_t numWires = threadPairCount / kWires + ((threadPairCount % kWires) ? 1 : 0);

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
        auto threadPairs = assign_thread_affinity_pairs(threadPairCount);

        // ---- Producers: read and decode TIFF slices ----
        for (size_t t = 0; t < threadPairCount; ++t) {
            BoundedQueue<TaskPtr>& queueForPair = *queuesForWires[t / kWires];

            producerThreads.emplace_back([&, t] {
                set_thread_affinity(threadPairs[t].producerLogicalCore);
                std::vector<uint8_t> tempBuf;  // Thread-local decode buffer

                while (true) {
                    if (abortFlag.load(std::memory_order_acquire)) break;
                    uint32_t idx = nextSliceIndex.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= numSlices) break;

                    const auto& task = tasks[idx];
                    try {
                        TiffHandle tif(TIFFOpen(task.path.c_str(), "r"));
                        if (!tif) {
                            std::lock_guard<std::mutex> lck(errorMutex);
                            runtimeErrors.emplace_back("Slice " + std::to_string(task.zIndex + 1) +
                                                      ": Cannot open file " + task.path);
                            abortFlag.store(true, std::memory_order_release);
                            break;
                        }
                        // Create a TaskResult with decoded data
                        auto res = std::make_shared<TaskResult>(idx, static_cast<size_t>(task.cropH * task.cropW * bytesPerPixel), task.cropH, task.cropW);
                        readSubRegionToBuffer(task, tif.get(), bytesPerPixel, res->data, tempBuf);
                        queueForPair.push(res);

                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lck(errorMutex);
                        runtimeErrors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": " + ex.what());
                        abortFlag.store(true, std::memory_order_release);
                        break;
                    } catch (...) {
                        std::lock_guard<std::mutex> lck(errorMutex);
                        runtimeErrors.emplace_back("Slice " + std::to_string(task.zIndex + 1) + ": Unknown exception");
                        abortFlag.store(true, std::memory_order_release);
                        break;
                    }
                }
                // Signal end-of-tasks to consumer(s)
                queueForPair.push(nullptr);
            });
        }

        // ---- Consumers: copy into output buffer (with optional transpose) ----
        for (size_t t = 0; t < threadPairCount; ++t) {
            BoundedQueue<TaskPtr>& queueForPair = *queuesForWires[t / kWires];

            consumerThreads.emplace_back([&, t] {
                set_thread_affinity(threadPairs[t].consumerLogicalCore);
                while (true) {
                    if (abortFlag.load(std::memory_order_acquire)) break;
                    TaskPtr res;
                    queueForPair.wait_and_pop(res);
                    if (!res) break; // End-of-tasks signal

                    const auto& task = tasks[res->block_id];
                    bool is_full_frame = !task.transpose &&
                                         task.in_row0 == 0 && task.in_col0 == 0 &&
                                         task.cropH == task.roiH && task.cropW == task.roiW;

                    if (is_full_frame) {
                        // Fast: entire slice is contiguous, just copy block
                        size_t dstByte = task.zIndex * task.pixelsPerSlice * bytesPerPixel;
                        size_t sliceBytes = task.pixelsPerSlice * bytesPerPixel;
                        std::memcpy(static_cast<uint8_t*>(outData) + dstByte,
                                    res->data.data(),
                                    sliceBytes);
                    } else {
                        // General case (slower, but always correct)
                        for (uint32_t row = 0; row < task.cropH; ++row) {
                            for (uint32_t col = 0; col < task.cropW; ++col) {
                                size_t dstElem = computeDstIndex(task, row, col);
                                size_t dstByte = dstElem * bytesPerPixel;
                                size_t srcByte = (row * task.cropW + col) * bytesPerPixel;
                                std::memcpy(static_cast<uint8_t*>(outData) + dstByte,
                                            res->data.data() + srcByte,
                                            bytesPerPixel);
                            }
                        }
                    }
                }
            });
        }

        // ---- Wait for all threads ----
        for (auto& p : producerThreads) p.join();
        for (auto& c : consumerThreads) c.join();

        if (!runtimeErrors.empty()) {
            std::ostringstream allerr;
            allerr << "Errors during load_bl_tif (producer/consumer):\n";
            for (const auto& s : runtimeErrors) {
                allerr << "  - " << s << "\n";
            }
            mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", allerr.str().c_str());
        }
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_bl_tif:Error", "%s", ex.what());
    }
}
