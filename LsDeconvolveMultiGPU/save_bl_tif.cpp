/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads]);

  INPUTS:
    • volume        : 3D MATLAB array (uint8 or uint16).
    • fileList      : 1×Z cell array of output filenames.
    • isXYZ         : scalar logical/numeric. True if array is [X Y Z].
    • compression   : "none", "lzw", or "deflate".
    • nThreads      : (opt) max threads (default = half hardware_concurrency).

  FEATURES:
    • Guard-clauses on invalid/read-only output paths.
    • Atomic slice-index dispatch (chunks of slices).
    • Direct strip-write when input is XYZ (row-major) to avoid buffers.
    • Per-thread lazy strip buffer for YX inputs.
    • Multi-row strips (64 rows) tuned for deflate compression.
    • Deflate predictor & quality tuned for best libtiff 4.7 compression.
    • Safe temp-file → rename to avoid NUMA lockups.
    • C++17 <filesystem>, RAII for TIFF handles.
    • Exception aggregation/reporting in main thread.

  LIMITATIONS:
    • Grayscale only (single channel).
    • Single-strip file per strip group.

  DEPENDENCIES:
    • libtiff ≥ 4.7, MATLAB MEX API, C++17 <filesystem>.

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
    if (width >= 1024 && height >= 1024) {
        tileWidth = 1024; tileLength = 1024;
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
    size_t               sliceIdx,
    size_t               widthDim,
    size_t               heightDim,
    size_t               bytesPerPixel,
    bool                 isXYZ,
    uint16_t             compressionType,
    const std::string&   outputPath,
    bool                 useTiles             // <--- new
) {
    // widthDim = pixels per row, heightDim = number of rows
    const uint32_t imageWidth  = static_cast<uint32_t>(widthDim);
    const uint32_t imageHeight = static_cast<uint32_t>(heightDim);
    const size_t   sliceSize   = widthDim * heightDim * bytesPerPixel;
    const uint8_t* basePtr     = volumeData + sliceIdx * sliceSize;

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
            const size_t   tileBytes    = size_t(tileWidth) * tileLength * bytesPerPixel;

            // No change to buffer logic: isXYZ / else
            if (isXYZ) {
                for (uint32_t td = 0; td < tilesDown; ++td) {
                    for (uint32_t ta = 0; ta < tilesAcross; ++ta) {
                        // Determine region bounds
                        uint32_t rowStart    = td * tileLength;
                        uint32_t rowsToWrite = std::min(tileLength, imageHeight - rowStart);
                        uint32_t colStart    = ta * tileWidth;
                        uint32_t colsToWrite = std::min(tileWidth,  imageWidth  - colStart);

                        // Allocate tile buffer
                        std::vector<uint8_t> tileBuffer(tileBytes, 0);

                        // Copy region into tileBuffer
                        for (uint32_t r = 0; r < rowsToWrite; ++r) {
                            size_t srcOffset = (rowStart + r) * imageWidth * bytesPerPixel + colStart * bytesPerPixel;
                            size_t dstOffset = r * tileWidth * bytesPerPixel;
                            std::memcpy(&tileBuffer[dstOffset], basePtr + srcOffset, colsToWrite * bytesPerPixel);
                        }
                        // Write tile (incomplete tiles are zero-padded)
                        tstrip_t tileIdx = TIFFComputeTile(tif, colStart, rowStart, 0, 0);
                        if (TIFFWriteEncodedTile(tif, tileIdx, tileBuffer.data(), tileBytes) < 0)
                            throw std::runtime_error("TIFF tile write failed at (" + std::to_string(ta) + "," + std::to_string(td) + ")");
                    }
                }
            } else {
                // column-major to row-major tile
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
                                size_t srcOffset = ((rowStart + r) + (colStart + c)*heightDim) * bytesPerPixel;
                                size_t dstOffset = (r * tileWidth + c) * bytesPerPixel;
                                std::memcpy(&tileBuffer[dstOffset], basePtr + srcOffset, bytesPerPixel);
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
                    uint32_t rowsToWrite = std::min<uint32_t>(rowsPerStrip, imageHeight - rowStart);
                    size_t   byteCount   = size_t(imageWidth) * rowsToWrite * bytesPerPixel;
                    const uint8_t* dataPtr = basePtr + size_t(rowStart) * imageWidth * bytesPerPixel;
                    void* buf = const_cast<void*>(static_cast<const void*>(dataPtr));
                    if (TIFFWriteEncodedStrip(tif, s, buf, byteCount) < 0)
                        throw std::runtime_error("TIFF write failed on strip " + std::to_string(s));
                }
            } else {
                thread_local std::vector<uint8_t> stripBuffer;
                for (uint32_t s = 0; s < numStrips; ++s) {
                    uint32_t rowStart    = s * rowsPerStrip;
                    uint32_t rowsToWrite = std::min<uint32_t>(rowsPerStrip, imageHeight - rowStart);
                    size_t   byteCount   = size_t(imageWidth) * rowsToWrite * bytesPerPixel;
                    if (stripBuffer.size() < byteCount)
                        stripBuffer.resize(byteCount);
                    for (uint32_t r = 0; r < rowsToWrite; ++r) {
                        for (uint32_t c = 0; c < imageWidth; ++c) {
                            size_t srcOff = ((rowStart + r) + c*heightDim) * bytesPerPixel;
                            size_t dstOff = (r*imageWidth  + c)      * bytesPerPixel;
                            std::memcpy(&stripBuffer[dstOff], basePtr + srcOff, bytesPerPixel);
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
// save_bl_tif(volume, fileList, isXYZ, compression[, nThreads, useTiles])
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs < 4 || nrhs > 6)
            mexErrMsgIdAndTxt("save_bl_tif:usage",
                "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads, useTiles]);");

        if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
            mexErrMsgIdAndTxt("save_bl_tif:type", "Volume must be uint8 or uint16.");

        // Raw dims from array
        const mwSize* rawDims = mxGetDimensions(prhs[0]);
        const size_t   rawRows = rawDims[0];
        const size_t   rawCols = rawDims[1];
        const size_t   numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3 ? rawDims[2] : 1);

        // isXYZ flag
        const bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

        // Determine width (X) and height (Y) consistently
        const size_t widthDim  = isXYZ ? rawRows : rawCols;
        const size_t heightDim = isXYZ ? rawCols : rawRows;

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
        for (size_t i = 0; i < numSlices; ++i) {
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
        const size_t   bytesPerPixel = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2 : 1);

        // Thread count
        const size_t hwCores     = std::thread::hardware_concurrency();
        const size_t safeCores   = hwCores ? hwCores : 1;
        const size_t defaultTh   = std::max(safeCores / 2, size_t(1));
        const size_t reqTh       = (nrhs >= 5 ? static_cast<size_t>(mxGetScalar(prhs[4])) : defaultTh);
        const size_t threadCount = std::min(reqTh, numSlices);

        // Tiled mode (new): default off
        const bool useTiles = (nrhs >= 6) ? (mxIsLogicalScalarTrue(prhs[5]) || (mxIsNumeric(prhs[5]) && mxGetScalar(prhs[5]) != 0)) : false;

        // Error collection
        std::vector<std::string> errors;
        std::mutex              errorMutex;
        std::atomic<size_t>     nextSlice{0};

        // Launch workers
        std::vector<std::thread> workers;
        workers.reserve(threadCount);
        for (size_t thread_idx = 0; thread_idx < threadCount; ++thread_idx) {
            workers.emplace_back([&, thread_idx]() {
                set_thread_affinity(thread_idx);
                while (true) {
                    size_t start = nextSlice.fetch_add(slicesPerDispatch, std::memory_order_relaxed);
                    if (start >= numSlices) break;
                    size_t end = std::min(numSlices, start + slicesPerDispatch);
                    for (size_t idx = start; idx < end; ++idx) {
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
