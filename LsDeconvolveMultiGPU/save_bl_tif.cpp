/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.
  Supports both STRIP (default) and TILE (optional) TIFF writing modes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads, useTiles]);

  INPUTS:
    • volume      : 3D MATLAB array (uint8 or uint16), or 2D for single slice.
    • fileList    : 1×Z cell array of output filenames, one per Z-slice.
    • isXYZ       : Scalar logical/numeric. True if 'volume' is [X Y Z], false if [Y X Z].
    • compression : String. "none", "lzw", or "deflate".
    • nThreads    : (Optional) Number of threads to use. Default = half hardware concurrency. Pass [] to auto-select.
    • useTiles    : (Optional) true to use tiled TIFF output (TIFFWriteEncodedTile), false for classic strip mode (TIFFWriteEncodedStrip, default).

  FEATURES:
    • Multi-threaded, atomic slice dispatch for maximum throughput.
    • Direct strip/tile write when input is row-major (isXYZ==true); efficient buffer conversion otherwise.
    • Automatic tile size selection (1024×1024 for large images, 256×256 for small).
    • Guard-clauses on invalid or read-only output paths.
    • Per-thread affinity setting for improved NUMA balancing on multi-socket systems.
    • Safe temp-file → rename to prevent partial writes or NUMA lockups.
    • Exception aggregation and robust error reporting to MATLAB.

  NOTES:
    • Grayscale only (single channel per slice).
    • Writes each Z-slice to a separate TIFF file; does NOT create multi-page TIFFs.
    • "useTiles" is for performance/compatibility; for most scientific image stacks, STRIP mode is usually faster.
    • Compression "deflate" uses predictor and modest quality for best libtiff performance.

  EXAMPLE:
    % Save a 3D [X Y Z] volume as LZW-compressed TIFFs, auto threads, STRIP mode:
    save_bl_tif(vol, fileList, true, 'lzw');

    % Save with explicit 8 threads, in TILE mode:
    save_bl_tif(vol, fileList, true, 'deflate', 8, true);

  DEPENDENCIES:
    • libtiff ≥ 4.7, MATLAB MEX API, C++17 <filesystem>, POSIX/Windows threading.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)

  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <system_error>
#include <cstdint>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
  // prevent <windows.h> from defining min/max as macros
  #ifndef NOMINMAX
  #  define NOMINMAX
  #endif
  #include <windows.h>
  #include <io.h>
  #ifndef W_OK
    #define W_OK 2
  #endif
  #define access _access
#elif defined(__linux__)
  #include <sched.h>
  #include <pthread.h>
  #include <unistd.h>
#else
  #include <unistd.h>
#endif

namespace fs = std::filesystem;

// Set thread affinity for best NUMA/core balancing
inline void set_thread_affinity(size_t thread_idx) {
#if defined(_WIN32)
    DWORD num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 1;
    DWORD_PTR mask = (1ull << (thread_idx % num_cores));
    HANDLE hThread = GetCurrentThread();
    SetThreadAffinityMask(hThread, mask);
#elif defined(__linux__)
    unsigned num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 1;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_idx % num_cores, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)thread_idx;
#endif
}

// Number of rows per TIFF strip (for RAM vs compression balance)
static constexpr uint32_t rowsPerStrip = 64;
// Number of slices claimed per atomic dispatch
static constexpr size_t slicesPerDispatch = 4;

// New: tile size selection logic (called only for tile mode)
inline void select_tile_size(uint32_t width, uint32_t height, uint32_t &tileWidth, uint32_t &tileLength) {
    const uint32_t long_length = 2048;
    if (width >= long_length && height >= long_length) {
        tileWidth = long_length; tileLength = long_length;
    } else {
        tileWidth = 256; tileLength = 256;
    }
}

// RAII wrapper for TIFF*
struct TiffWriter {
    TIFF* tif;
    TiffWriter(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffWriter() { if (tif) TIFFClose(tif); }
};

// Write a single Z-slice to TIFF: strip or tile mode based on flag
static void writeSliceToTiff(
    const uint8_t*       volumeData,
    uint32_t             sliceIdx,
    uint32_t             widthDim,
    uint32_t             heightDim,
    uint32_t             bytesPerPixel,
    bool                 isXYZ,
    uint16_t             compressionType,
    const std::string&   outputPath,
    bool                 useTiles
) {
    // widthDim = pixels per row, heightDim = number of rows
    const uint32_t imageWidth  = widthDim;
    const uint32_t imageHeight = heightDim;
    const size_t   sliceSize   = size_t(widthDim) * size_t(heightDim) * size_t(bytesPerPixel);
    const uint8_t* basePtr     = volumeData + size_t(sliceIdx) * sliceSize;

    fs::path tempFile = fs::path(outputPath).concat(".tmp");

    {
        TiffWriter writer(tempFile.string(), "w");
        TIFF* tif = writer.tif;

        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      imageWidth);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     imageHeight);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   static_cast<uint16_t>(bytesPerPixel * 8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
        TIFFSetField(tif, TIFFTAG_COMPRESSION,     compressionType);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

        // DEFLATE: modest compression level (1–9)
        if (compressionType == COMPRESSION_ADOBE_DEFLATE) {
            const int zipLevel = 1;
            TIFFSetField(tif, TIFFTAG_ZIPQUALITY, zipLevel);
            TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
        }

        if (useTiles) {
            // ----------- TILED MODE -----------
            uint32_t tileWidth, tileLength;
            select_tile_size(imageWidth, imageHeight, tileWidth, tileLength);
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, tileWidth);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, tileLength);

            const uint32_t tilesAcross  = (imageWidth  + tileWidth  - 1) / tileWidth;
            const uint32_t tilesDown    = (imageHeight + tileLength - 1) / tileLength;
            const size_t   tileBytes    = size_t(tileWidth) * size_t(tileLength) * size_t(bytesPerPixel);

            if (isXYZ) {
                for (uint32_t td = 0; td < tilesDown; ++td) {
                    for (uint32_t ta = 0; ta < tilesAcross; ++ta) {
                        uint32_t rowStart    = td * tileLength;
                        uint32_t rowsToWrite = std::min(tileLength, imageHeight - rowStart);
                        uint32_t colStart    = ta * tileWidth;
                        uint32_t colsToWrite = std::min(tileWidth,  imageWidth  - colStart);

                        std::vector<uint8_t> tileBuffer(tileBytes, 0);

                        for (uint32_t r = 0; r < rowsToWrite; ++r) {
                            size_t srcOffset = (size_t(rowStart) + size_t(r)) * size_t(imageWidth) * size_t(bytesPerPixel)
                                             + size_t(colStart) * size_t(bytesPerPixel);
                            size_t dstOffset = size_t(r) * size_t(tileWidth) * size_t(bytesPerPixel);
                            std::memcpy(&tileBuffer[dstOffset], basePtr + srcOffset, size_t(colsToWrite) * size_t(bytesPerPixel));
                        }
                        tstrip_t tileIdx = TIFFComputeTile(tif, colStart, rowStart, 0, 0);
                        if (TIFFWriteEncodedTile(tif, tileIdx, tileBuffer.data(), tileBytes) < 0)
                            throw std::runtime_error("TIFF tile write failed at (" + std::to_string(ta) + "," + std::to_string(td) + ")");
                    }
                }
            } else {
                thread_local std::vector<uint8_t> tileBuffer;
                if (tileBuffer.size() < tileBytes) tileBuffer.resize(tileBytes, 0);
                for (uint32_t td = 0; td < tilesDown; ++td) {
                    for (uint32_t ta = 0; ta < tilesAcross; ++ta) {
                        uint32_t rowStart    = td * tileLength;
                        uint32_t rowsToWrite = std::min(tileLength, imageHeight - rowStart);
                        uint32_t colStart    = ta * tileWidth;
                        uint32_t colsToWrite = std::min(tileWidth,  imageWidth  - colStart);
                        std::fill(tileBuffer.begin(), tileBuffer.end(), 0);
                        for (uint32_t r = 0; r < rowsToWrite; ++r) {
                            for (uint32_t c = 0; c < colsToWrite; ++c) {
                                size_t srcOffset = (size_t(rowStart + r) + size_t(colStart + c) * size_t(heightDim)) * size_t(bytesPerPixel);
                                size_t dstOffset = (size_t(r) * size_t(tileWidth) + size_t(c)) * size_t(bytesPerPixel);
                                std::memcpy(&tileBuffer[dstOffset], basePtr + srcOffset, size_t(bytesPerPixel));
                            }
                        }
                        tstrip_t tileIdx = TIFFComputeTile(tif, colStart, rowStart, 0, 0);
                        if (TIFFWriteEncodedTile(tif, tileIdx, tileBuffer.data(), tileBytes) < 0)
                            throw std::runtime_error("TIFF tile write failed at (" + std::to_string(ta) + "," + std::to_string(td) + ")");
                    }
                }
            }
        } else {
            // ----------- STRIPED MODE -----------
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, std::min(rowsPerStrip, imageHeight));
            const uint32_t numStrips = (imageHeight + rowsPerStrip - 1) / rowsPerStrip;
            if (isXYZ) {
                for (uint32_t s = 0; s < numStrips; ++s) {
                    uint32_t rowStart    = s * rowsPerStrip;
                    uint32_t rowsToWrite = std::min(rowsPerStrip, imageHeight - rowStart);
                    size_t   byteCount   = size_t(imageWidth) * size_t(rowsToWrite) * size_t(bytesPerPixel);
                    const uint8_t* dataPtr = basePtr + size_t(rowStart) * size_t(imageWidth) * size_t(bytesPerPixel);
                    void* buf = const_cast<void*>(static_cast<const void*>(dataPtr));
                    if (TIFFWriteEncodedStrip(tif, s, buf, byteCount) < 0)
                        throw std::runtime_error("TIFF write failed on strip " + std::to_string(s));
                }
            } else {
                thread_local std::vector<uint8_t> stripBuffer;
                for (uint32_t s = 0; s < numStrips; ++s) {
                    uint32_t rowStart    = s * rowsPerStrip;
                    uint32_t rowsToWrite = std::min(rowsPerStrip, imageHeight - rowStart);
                    size_t   byteCount   = size_t(imageWidth) * size_t(rowsToWrite) * size_t(bytesPerPixel);
                    if (stripBuffer.size() < byteCount)
                        stripBuffer.resize(byteCount);
                    for (uint32_t r = 0; r < rowsToWrite; ++r) {
                        for (uint32_t c = 0; c < imageWidth; ++c) {
                            size_t srcOff = (size_t(rowStart + r) + size_t(c) * size_t(heightDim)) * size_t(bytesPerPixel);
                            size_t dstOff = (size_t(r) * size_t(imageWidth)  + size_t(c)) * size_t(bytesPerPixel);
                            std::memcpy(&stripBuffer[dstOff], basePtr + srcOff, size_t(bytesPerPixel));
                        }
                    }
                    if (TIFFWriteEncodedStrip(tif, s, stripBuffer.data(), byteCount) < 0)
                        throw std::runtime_error("TIFF write failed on strip " + std::to_string(s));
                }
            }
        }
    }

    // Atomic replace
    std::error_code ec;
    fs::rename(tempFile, outputPath, ec);
    if (ec) {
        if (fs::exists(outputPath)) fs::remove(outputPath);
        fs::rename(tempFile, outputPath, ec);
        if (ec)
            throw std::runtime_error("Failed to rename " + tempFile.string() + " → " + outputPath);
    }
}

// MATLAB MEX entry point
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 6)
            mexErrMsgIdAndTxt("save_bl_tif:usage",
                "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads, useTiles]);");

        if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
            mexErrMsgIdAndTxt("save_bl_tif:type", "Volume must be uint8 or uint16.");

        // Raw dims from array
        const mwSize*  rawDims   = mxGetDimensions(prhs[0]);
        const uint32_t rawRows   = static_cast<uint32_t>(rawDims[0]);
        const uint32_t rawCols   = static_cast<uint32_t>(rawDims[1]);
        const uint32_t numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3 ? static_cast<uint32_t>(rawDims[2]) : 1);

        // isXYZ flag
        const bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        // Determine width (X) and height (Y) consistently
        const uint32_t widthDim  = isXYZ ? rawRows : rawCols;
        const uint32_t heightDim = isXYZ ? rawCols : rawRows;

        // Compression
        char* compCStr = mxArrayToUTF8String(prhs[3]);
        std::string compressionStr(compCStr);
        mxFree(compCStr);
        uint16_t compressionType =
               compressionStr == "none"    ? COMPRESSION_NONE
             : compressionStr == "lzw"     ? COMPRESSION_LZW
             : compressionStr == "deflate" ? COMPRESSION_ADOBE_DEFLATE
             : throw std::runtime_error("Invalid compression: " + compressionStr);

        // fileList validation
        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numSlices)
            mexErrMsgIdAndTxt("save_bl_tif:files",
                "fileList must be a cell-array of length = number of slices.");
        std::vector<std::string> outputPaths(numSlices);
        for (uint32_t i = 0; i < numSlices; ++i) {
            char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
            outputPaths[i] = s;
            mxFree(s);
        }

        // Guard-clauses
        for (auto& path : outputPaths) {
            fs::path dir = fs::path(path).parent_path();
            if (!dir.empty() && !fs::exists(dir))
                mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                    "Directory does not exist: %s", dir.string().c_str());
            if (fs::exists(path) && access(path.c_str(), W_OK) != 0)
                mexErrMsgIdAndTxt("save_bl_tif:readonly",
                    "Cannot overwrite read-only file: %s", path.c_str());
        }

        // Data pointer & sample size
        const uint8_t* volumeData    = static_cast<const uint8_t*>(mxGetData(prhs[0]));
        const uint32_t bytesPerPixel = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2u : 1u);

        // Thread count
        const size_t hwCores     = std::thread::hardware_concurrency();
        const size_t safeCores   = hwCores ? hwCores : 1;
        const size_t defaultTh   = std::max(safeCores / 2, size_t(1));
        const size_t reqTh       = (nrhs >= 5 && !mxIsEmpty(prhs[4])? static_cast<size_t>(mxGetScalar(prhs[4])) : defaultTh);
        const size_t threadCount = std::min(reqTh, static_cast<size_t>(numSlices));

        // Tiled mode (new): default off
        const bool useTiles = (nrhs >= 6) ? (mxIsLogicalScalarTrue(prhs[5]) || (mxIsNumeric(prhs[5]) && mxGetScalar(prhs[5]) != 0)) : false;

        // Error collection
        std::vector<std::string> errors;
        std::mutex              errorMutex;
        std::atomic<uint32_t>     nextSlice{0};

        // Launch workers
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        for (size_t thread_idx = 0; thread_idx < threadCount; ++thread_idx) {
            workers.emplace_back([&, thread_idx]() {
                //set_thread_affinity(thread_idx);
                while (true) {
                    uint32_t start = nextSlice.fetch_add(static_cast<uint32_t>(slicesPerDispatch), std::memory_order_relaxed);
                    if (start >= numSlices) break;
                    uint32_t end = std::min(numSlices, start + static_cast<uint32_t>(slicesPerDispatch));
                    for (uint32_t idx = start; idx < end; ++idx) {
                        try {
                            writeSliceToTiff(volumeData, idx, widthDim, heightDim, bytesPerPixel, isXYZ, compressionType, outputPaths[idx], useTiles);
                        } catch (const std::exception& ex) {
                            std::lock_guard<std::mutex> lg(errorMutex);
                            errors.push_back(ex.what());
                            return;
                        }
                    }
                }
            });
        }

        for (auto& th : workers) th.join();

        if (!errors.empty())
            mexErrMsgIdAndTxt("save_bl_tif:runtime", errors.front().c_str());

        if (nlhs > 0)
            plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("save_bl_tif:runtime", ex.what());
    }
    catch (...) {
        mexErrMsgIdAndTxt("save_bl_tif:unknown", "Unknown error in save_bl_tif");
    }
}