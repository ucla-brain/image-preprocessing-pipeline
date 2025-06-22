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
#include <unistd.h>

namespace fs = std::filesystem;

// Number of rows per TIFF strip (for balance of RAM vs compression)
static constexpr uint32_t rowsPerStrip      = 64;
// Number of slices claimed per atomic dispatch
static constexpr size_t   slicesPerDispatch = 4;

// RAII wrapper for TIFF* so we never leak handles
struct TiffWriter {
    TIFF* tif;
    TiffWriter(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffWriter() {
        if (tif) TIFFClose(tif);
    }
};

// Write a single Z-slice to TIFF, using either a direct-write path
// (when isXYZ==true so memory is row-major) or a per-thread buffer otherwise.
static void writeSliceToTiff(
    const uint8_t*       volumeData,
    size_t               sliceIdx,
    size_t               widthDim,
    size_t               heightDim,
    size_t               bytesPerPixel,
    bool                 isXYZ,
    uint16_t             compressionType,
    const std::string&   outputPath
) {
    // Determine image dims for libtiff
    const uint32_t imageWidth  = isXYZ ? static_cast<uint32_t>(widthDim)
                                       : static_cast<uint32_t>(heightDim);
    const uint32_t imageHeight = isXYZ ? static_cast<uint32_t>(heightDim)
                                       : static_cast<uint32_t>(widthDim);
    const size_t   sliceSize   = widthDim * heightDim * bytesPerPixel;
    const uint8_t* basePtr     = volumeData + sliceIdx * sliceSize;

    // Build temporary filename
    fs::path tempFile = fs::path(outputPath).concat(".tmp");

    // Open and set TIFF tags
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
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,    rowsPerStrip);

        // If deflate, tune predictor & quality
        if (compressionType == COMPRESSION_DEFLATE) {
            TIFFSetField(tif, TIFFTAG_PREDICTOR,  PREDICTOR_HORIZONTAL);
            TIFFSetField(tif, TIFFTAG_ZIPQUALITY, 6);  // default 1–9
        }

        const uint32_t numStrips = (imageHeight + rowsPerStrip - 1) / rowsPerStrip;

        // Per-thread buffer for YX layouts
        thread_local std::vector<uint8_t> stripBuffer;

        for (uint32_t stripIndex = 0; stripIndex < numStrips; ++stripIndex) {
            const uint32_t rowStart     = stripIndex * rowsPerStrip;
            const uint32_t rowsToWrite  = std::min<uint32_t>(rowsPerStrip,
                                              imageHeight - rowStart);
            const size_t   byteCount    = static_cast<size_t>(imageWidth)
                                          * rowsToWrite * bytesPerPixel;

            if (isXYZ) {
                // Direct-write path: memory is row-major in X (widthDim)
                const uint8_t* stripData = basePtr +
                    static_cast<size_t>(rowStart) * imageWidth * bytesPerPixel;
                // cast away const so libtiff accepts void*
                void* dataPtr = const_cast<void*>(
                                  static_cast<const void*>(stripData));
                tsize_t wrote = (compressionType == COMPRESSION_NONE)
                    ? TIFFWriteRawStrip    (tif, stripIndex, dataPtr, byteCount)
                    : TIFFWriteEncodedStrip(tif, stripIndex, dataPtr, byteCount);
                if (wrote < 0)
                    throw std::runtime_error("TIFF write failed on strip " +
                                             std::to_string(stripIndex));
            } else {
                // YX layout: pack into contiguous buffer
                if (stripBuffer.size() < byteCount)
                    stripBuffer.resize(byteCount);
                for (uint32_t r = 0; r < rowsToWrite; ++r) {
                    for (uint32_t c = 0; c < imageWidth; ++c) {
                        // MATLAB is column-major: index = (row + col*heightDim)
                        size_t srcOffset = ((rowStart + r)
                                          + c * heightDim)
                                          * bytesPerPixel;
                        size_t dstOffset = (r * imageWidth + c)
                                          * bytesPerPixel;
                        std::memcpy(&stripBuffer[dstOffset],
                                    basePtr + srcOffset,
                                    bytesPerPixel);
                    }
                }
                // stripBuffer.data() is non-const, so no cast needed
                tsize_t wrote = (compressionType == COMPRESSION_NONE)
                    ? TIFFWriteRawStrip    (tif, stripIndex, stripBuffer.data(), byteCount)
                    : TIFFWriteEncodedStrip(tif, stripIndex, stripBuffer.data(), byteCount);
                if (wrote < 0)
                    throw std::runtime_error("TIFF write failed on strip " +
                                             std::to_string(stripIndex));
            }
        }
    } // TIFFClose happens here

    // Atomically replace any existing file
    std::error_code ec;
    fs::rename(tempFile, outputPath, ec);
    if (ec) {
        if (fs::exists(outputPath)) fs::remove(outputPath);
        fs::rename(tempFile, outputPath, ec);
        if (ec)
            throw std::runtime_error("Failed to rename " + tempFile.string() +
                                     " → " + outputPath);
    }
}

// MATLAB MEX entry point
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // 1) Check argument count
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:usage",
            "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");

    // 2) Validate volume type
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
        mexErrMsgIdAndTxt("save_bl_tif:type",
            "Volume must be uint8 or uint16.");

    // 3) Extract dimensions
    const mwSize* dims      = mxGetDimensions(prhs[0]);
    const size_t   heightDim = dims[0];
    const size_t   widthDim  = dims[1];
    const size_t   numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3
                                ? dims[2] : 1);

    // 4) Parse isXYZ flag
    const bool isXYZ = mxIsLogicalScalarTrue(prhs[2])
        || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // 5) Parse compression string
    char* compCStr = mxArrayToUTF8String(prhs[3]);
    std::string compressionStr(compCStr);
    mxFree(compCStr);
    uint16_t compressionType =
           compressionStr == "none"    ? COMPRESSION_NONE
         : compressionStr == "lzw"     ? COMPRESSION_LZW
         : compressionStr == "deflate" ? COMPRESSION_DEFLATE
         : throw std::runtime_error("Invalid compression: " + compressionStr);

    // 6) Validate fileList
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numSlices)
        mexErrMsgIdAndTxt("save_bl_tif:files",
            "fileList must be a cell-array of length = number of slices.");
    std::vector<std::string> outputFilePaths(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        outputFilePaths[i] = s;
        mxFree(s);
    }

    // 7) Guard-clauses: directories exist & writable
    for (auto& path : outputFilePaths) {
        fs::path dir = fs::path(path).parent_path();
        if (!dir.empty() && !fs::exists(dir))
            mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                "Directory does not exist: %s", dir.string().c_str());
        if (fs::exists(path) && access(path.c_str(), W_OK) != 0)
            mexErrMsgIdAndTxt("save_bl_tif:readonly",
                "Cannot overwrite read-only file: %s", path.c_str());
    }

    // 8) Get volume data pointer and sample size
    const uint8_t* volumeData     = static_cast<const uint8_t*>(mxGetData(prhs[0]));
    const size_t   bytesPerPixel  = (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2 : 1);

    // 9) Determine thread count
    const size_t hardwareConcurrency = std::thread::hardware_concurrency() ?: 1;
    const size_t defaultThreads      = std::max(hardwareConcurrency/2, size_t(1));
    const size_t requestedThreads    = (nrhs == 5
                                        ? static_cast<size_t>(mxGetScalar(prhs[4]))
                                        : defaultThreads);
    const size_t threadCount         = std::min(requestedThreads, numSlices);

    // 10) Prepare for error collection
    std::vector<std::string> errors;
    std::mutex              errorMutex;

    // 11) Atomic slice index
    std::atomic<size_t> nextSliceIndex{0};

    // 12) Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(threadCount);
    for (size_t t = 0; t < threadCount; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                const size_t start = nextSliceIndex.fetch_add(
                    slicesPerDispatch, std::memory_order_relaxed);
                if (start >= numSlices) break;
                const size_t end = std::min(numSlices, start + slicesPerDispatch);
                for (size_t sliceIdx = start; sliceIdx < end; ++sliceIdx) {
                    try {
                        writeSliceToTiff(volumeData,
                                         sliceIdx,
                                         widthDim,
                                         heightDim,
                                         bytesPerPixel,
                                         isXYZ,
                                         compressionType,
                                         outputFilePaths[sliceIdx]);
                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lock(errorMutex);
                        errors.push_back(ex.what());
                        return;
                    }
                }
            }
        });
    }

    // 13) Wait for all threads
    for (auto& th : threads) th.join();

    // 14) Report any errors
    if (!errors.empty())
        mexErrMsgIdAndTxt("save_bl_tif:runtime", errors.front().c_str());

    // 15) Optionally return the input volume
    if (nlhs > 0)
        plhs[0] = const_cast<mxArray*>(prhs[0]);
}
