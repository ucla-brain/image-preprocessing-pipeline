/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads]);

  INPUT:
    volume      : 3D MATLAB array of type uint8 or uint16.
    fileList    : 1×Z cell array of output filenames.
    isXYZ       : logical or numeric scalar. True if array is [X Y Z].
    compression : "none", "lzw", or "deflate".
    nThreads    : (optional) Number of threads (default = hardware concurrency).

  FEATURES:
    • Fixed-stride thread pool (no atomics).
    • Per-thread scratch buffer for row-major reordering.
    • RAII for TIFF handles.
    • Thread-safe error aggregation.
    • Correct orientation for both YXZ and XYZ.

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

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <sstream>
#include <unistd.h>  // for access()

// RAII wrapper for TIFF*
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

// Ensure we can overwrite or create the output file
static void ensureWritable(const std::string& path) {
    if (access(path.c_str(), F_OK) == 0 &&
        access(path.c_str(), W_OK) != 0) {
        throw std::runtime_error("Cannot overwrite read-only file: " + path);
    }
}

// Write one Z-slice: reorders into row-major then writes as one strip.
static void writeSlice(
    const uint8_t* baseData,
    size_t sliceIndex,
    size_t dim0,
    size_t dim1,
    size_t bytesPerSample,
    bool isXYZ,
    uint16_t compression,
    const std::string& outPath,
    std::vector<uint8_t>& scratch
) {
    // Determine image dimensions:
    //   if isXYZ, dims are [X Y], so height=Y=dim1, width=X=dim0
    //   else      dims are [Y X], so height=Y=dim0, width=X=dim1
    size_t height = isXYZ ? dim1 : dim0;
    size_t width  = isXYZ ? dim0 : dim1;
    size_t sliceBytes = dim0 * dim1 * bytesPerSample;

    // Pointers
    const uint8_t* src = baseData + sliceIndex * sliceBytes;

    // Prepare output file
    ensureWritable(outPath);
    std::string tmp = outPath + ".tmp";
    TiffHandle tiff(tmp, "w");
    TIFF* tif = tiff.tif;

    // TIFF tags
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bytesPerSample * 8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);

    // Reorder into row-major: scratch must be size = width*height*bytesPerSample
    // Mapping: for each image row r∈[0,height), col c∈[0,width),
    //   origRow = isXYZ ? c : r;
    //   origCol = isXYZ ? r : c;
    //   src offset = (origRow + origCol*dim0) * bytesPerSample
    //   dst offset = (r*width + c) * bytesPerSample
    uint8_t* dstPtr = scratch.data();
    for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
            size_t orow = isXYZ ? c : r;
            size_t ocol = isXYZ ? r : c;
            size_t srcOff = (orow + ocol * dim0) * bytesPerSample;
            size_t dstOff = (r    + c    * height) * bytesPerSample;
            // Note: dstOff uses dst row-major indexing via r*width + c
            dstOff = (r * width + c) * bytesPerSample;
            std::memcpy(dstPtr + dstOff,
                        src    + srcOff,
                        bytesPerSample);
        }
    }

    // Write the entire slice in one strip
    tsize_t written = (compression == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif, 0, scratch.data(), sliceBytes)
        : TIFFWriteEncodedStrip(tif, 0, scratch.data(), sliceBytes);
    if (written < 0)
        throw std::runtime_error("TIFF write failed for slice " +
                                 std::to_string(sliceIndex));

    // Atomic replace
    std::remove(outPath.c_str());
    if (std::rename(tmp.c_str(), outPath.c_str()) != 0)
        throw std::runtime_error("Failed to rename slice " +
                                 std::to_string(sliceIndex));
}

// Entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:usage",
          "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");

    // Validate type
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
        mexErrMsgIdAndTxt("save_bl_tif:type",
          "Volume must be uint8 or uint16.");

    // Dimensions
    const mwSize* dims = mxGetDimensions(prhs[0]];
    size_t dim0 = dims[0], dim1 = dims[1];
    size_t numSlices = (mxGetNumberOfDimensions(prhs[0])==3)
                       ? dims[2] : 1;

    // isXYZ flag
    bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                 (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // Compression
    char* compStr = mxArrayToUTF8String(prhs[3]);
    std::string cmp(compStr);
    mxFree(compStr);
    uint16_t compression =
         (cmp == "none")    ? COMPRESSION_NONE
       : (cmp == "lzw")     ? COMPRESSION_LZW
       : (cmp == "deflate") ? COMPRESSION_DEFLATE
       : throw std::runtime_error("Invalid compression: " + cmp);

    // File list
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != numSlices)
        mexErrMsgIdAndTxt("save_bl_tif:files",
          "fileList must have one entry per slice.");
    std::vector<std::string> files(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        files[i] = s;
        mxFree(s);
    }

    // Data pointer & sample size
    const uint8_t* volumeData =
      static_cast<const uint8_t*>(mxGetData(prhs[0]));
    size_t bytesPerSample =
      (mxGetClassID(prhs[0]) == mxUINT16_CLASS ? 2 : 1);

    // Thread count
    size_t hw = std::thread::hardware_concurrency();
    size_t req = (nrhs == 5) ? (size_t)mxGetScalar(prhs[4]) : hw;
    size_t numThreads = std::min(req, numSlices);
    if (numThreads < 1) numThreads = 1;

    // Prepare error collection
    std::vector<std::string> errors;
    std::mutex errLock;

    // Preallocate per-thread scratch
    std::vector<std::vector<uint8_t>> scratch(numThreads,
      std::vector<uint8_t>(dim0 * dim1 * bytesPerSample));

    // Launch threads (fixed-stride dispatch)
    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    for (size_t tid = 0; tid < numThreads; ++tid) {
        workers.emplace_back([&,tid]() {
            auto& buf = scratch[tid];
            for (size_t idx = tid; idx < numSlices; idx += numThreads) {
                try {
                    writeSlice(volumeData, idx,
                               dim0, dim1,
                               bytesPerSample,
                               isXYZ, compression,
                               files[idx],
                               buf);
                }
                catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lk(errLock);
                    errors.push_back(ex.what());
                }
            }
        });
    }

    // Join & report
    for (auto& t : workers) t.join();
    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "save_bl_tif errors:\n";
        for (auto& e : errors) oss << " - " << e << "\n";
        mexErrMsgIdAndTxt("save_bl_tif:runtime", oss.str().c_str());
    }

    if (nlhs > 0)  // return input if requested
        plhs[0] = const_cast<mxArray*>(prhs[0]);
}
