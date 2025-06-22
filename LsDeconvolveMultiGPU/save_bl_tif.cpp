/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads]);

  INPUT:
    volume      : 3D MATLAB array of type uint8 or uint16.
    fileList    : 1×Z cell array of output filenames.
    isXYZ       : logical or numeric scalar. True if volume is in [X Y Z] order.
    compression : "none", "lzw", or "deflate".
    nThreads    : (optional) Number of threads (default = hardware concurrency).

  FEATURES:
    • Fixed-size thread pool (threads reused).
    • Balanced slice assignment per thread (no atomics).
    • Memory scratch buffer only for transpose.
    • RAII for TIFF handles.
    • Thread-safe error aggregation.

  LIMITATIONS:
    • Grayscale only (single channel).
    • Single-strip TIFF per slice.
    • No retry on I/O errors.

  DEPENDENCIES:
    libtiff, MATLAB MEX API.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)

  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <sstream>
#include <unistd.h> // access()

// RAII wrapper for TIFF file handles
struct TiffHandle {
    TIFF* tif;
    TiffHandle(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif) throw std::runtime_error("Cannot open TIFF: " + path);
    }
    ~TiffHandle() {
        if (tif) TIFFClose(tif);
    }
};

// Ensure the output file is writable or does not exist
static void ensureWritable(const std::string& path) {
    if (access(path.c_str(), F_OK) == 0 &&
        access(path.c_str(), W_OK) != 0) {
        throw std::runtime_error("Cannot overwrite read-only file: " + path);
    }
}

// Write a single Z-slice to disk
static void writeSlice(
    const uint8_t* volumeData,
    size_t sliceIdx,
    size_t imgHeight,
    size_t imgWidth,
    size_t bytesPerSample,
    bool isXYZOrder,
    uint16_t compression,
    const std::string& outFile,
    std::vector<uint8_t>* transposeBuffer // nullptr if not needed
) {
    // Calculate byte offset and slice size
    size_t sliceSize = imgHeight * imgWidth * bytesPerSample;
    const uint8_t* srcPtr = volumeData + sliceIdx * sliceSize;

    ensureWritable(outFile);

    // Temporary path for atomic replace
    std::string tmpPath = outFile + ".tmp";

    // Open TIFF file
    TiffHandle tiff(tmpPath, "w");
    TIFF* tif = tiff.tif;

    // Set essential TIFF tags
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,
                 isXYZOrder ? imgWidth : imgHeight);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,
                 isXYZOrder ? imgHeight : imgWidth);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bytesPerSample * 8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,
                 isXYZOrder ? imgHeight : imgWidth);

    const uint8_t* writeBuf = srcPtr;

    // Transpose [Y X] → [X Y] if needed
    if (!isXYZOrder) {
        auto& buf = *transposeBuffer;
        for (size_t row = 0; row < imgHeight; ++row) {
            for (size_t col = 0; col < imgWidth; ++col) {
                size_t srcPos = (row * imgWidth + col) * bytesPerSample;
                size_t dstPos = (col * imgHeight + row) * bytesPerSample;
                std::memcpy(buf.data() + dstPos,
                            srcPtr + srcPos,
                            bytesPerSample);
            }
        }
        writeBuf = buf.data();
    }

    // Write raw or encoded strip
    tsize_t written = (compression == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif, 0, const_cast<uint8_t*>(writeBuf), sliceSize)
        : TIFFWriteEncodedStrip(tif, 0, const_cast<uint8_t*>(writeBuf), sliceSize);

    if (written < 0) {
        throw std::runtime_error("TIFF write failed for slice " +
                                 std::to_string(sliceIdx));
    }

    // Atomic file replace
    std::remove(outFile.c_str());
    if (std::rename(tmpPath.c_str(), outFile.c_str()) != 0) {
        throw std::runtime_error("Failed to rename slice " +
                                 std::to_string(sliceIdx));
    }
}

// MEX entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    // Validate input count
    if (nrhs < 4 || nrhs > 5) {
        mexErrMsgIdAndTxt("save_bl_tif:usage",
            "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");
    }
    // Validate data type
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0])) {
        mexErrMsgIdAndTxt("save_bl_tif:type",
            "Volume must be uint8 or uint16.");
    }

    // Get dimensions
    const mwSize* dims = mxGetDimensions(prhs[0]);
    size_t imgHeight = dims[0];
    size_t imgWidth  = dims[1];
    size_t numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3)
        ? dims[2] : 1;

    // Parse isXYZ flag
    bool isXYZOrder = mxIsLogicalScalarTrue(prhs[2]) ||
                      (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // Parse compression
    char* cStr = mxArrayToUTF8String(prhs[3]);
    std::string comp(cStr);
    mxFree(cStr);

    uint16_t compression;
    if      (comp == "none")    compression = COMPRESSION_NONE;
    else if (comp == "lzw")     compression = COMPRESSION_LZW;
    else if (comp == "deflate") compression = COMPRESSION_DEFLATE;
    else mexErrMsgIdAndTxt("save_bl_tif:compression",
             "Compression must be 'none', 'lzw', or 'deflate'.");

    // Parse fileList
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numSlices) {
        mexErrMsgIdAndTxt("save_bl_tif:files",
            "fileList must be a cell array matching number of slices.");
    }
    std::vector<std::string> fileList(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* f = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        fileList[i] = f;
        mxFree(f);
    }

    // Get raw volume data pointer and sample size
    const uint8_t* volumeData =
        static_cast<const uint8_t*>(mxGetData(prhs[0]));
    size_t bytesPerSample =
        (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2 : 1);

    // Determine thread count
    size_t hw = std::thread::hardware_concurrency();
    size_t req = (nrhs == 5)
        ? static_cast<size_t>(mxGetScalar(prhs[4]))
        : hw;
    size_t threadCount = std::min(req, numSlices);
    if (threadCount < 1) threadCount = 1;

    // Prepare error aggregation
    std::vector<std::string> errors;
    std::mutex errLock;

    // Preallocate transpose buffers if needed
    std::vector<std::vector<uint8_t>> scratchBufs;
    if (!isXYZOrder) {
        scratchBufs.resize(threadCount);
        size_t bufSize = imgHeight * imgWidth * bytesPerSample;
        for (auto& buf : scratchBufs) buf.resize(bufSize);
    }

    // Launch threads
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (size_t tid = 0; tid < threadCount; ++tid) {
        workers.emplace_back([&, tid]() {
            auto* scratch = !isXYZOrder ? &scratchBufs[tid] : nullptr;

            // Assign slices in fixed stride: tid, tid+N, tid+2N…
            for (size_t idx = tid; idx < numSlices; idx += threadCount) {
                try {
                    writeSlice(volumeData, idx,
                               imgHeight, imgWidth,
                               bytesPerSample,
                               isXYZOrder,
                               compression,
                               fileList[idx],
                               scratch);
                }
                catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lg(errLock);
                    errors.push_back(ex.what());
                }
            }
        });
    }

    // Join all threads
    for (auto& t : workers) t.join();

    // Report any errors
    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "Errors during save_bl_tif:\n";
        for (const auto& e : errors) oss << " - " << e << "\n";
        mexErrMsgIdAndTxt("save_bl_tif:runtime", oss.str().c_str());
    }

    // Return the input volume if requested
    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
}
