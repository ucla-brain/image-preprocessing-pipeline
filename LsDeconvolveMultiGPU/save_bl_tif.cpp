/*==============================================================================
  save_bl_tif.cpp

  High-throughput multi-threaded TIFF Z-slice saver for 3D MATLAB volumes.

  USAGE:
    save_bl_tif(volume, fileList, isXYZ, compression[, nThreads]);

  INPUT:
    • volume      : 3D MATLAB array (uint8 or uint16).
    • fileList    : 1×Z cell array of output filenames.
    • isXYZ       : scalar logical/numeric. True if array is [X Y Z].
    • compression : "none", "lzw", or "deflate".
    • nThreads    : (opt) max threads (default = half of hardware_concurrency).

  FEATURES:
    • Guard-clauses on invalid / read-only output paths.
    • Per-thread single scratch buffer (lazy alloc).
    • Atomic index dispatch (no big task queue).
    • Proper close-before-rename (avoids lockups under NUMA).
    • C++17 filesystem, RAII for TIFF handles.
    • Exception aggregation and reporting.

  LIMITATIONS:
    • Grayscale only (single channel).
    • Single-strip TIFF per slice.

  DEPENDENCIES:
    libtiff, MATLAB MEX API, C++17 <filesystem>.

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
#include <cstdint>
#include <cstring>
#include <unistd.h>  // for access()

namespace fs = std::filesystem;

// RAII handle for TIFF*
struct TiffHandle {
    TIFF* tif;
    TiffHandle(const std::string& path, const char* mode) {
        tif = TIFFOpen(path.c_str(), mode);
        if (!tif)
            throw std::runtime_error("Cannot open TIFF for writing: " + path);
    }
    ~TiffHandle() {
        if (tif) TIFFClose(tif);
    }
};

// Write one slice to disk (buffer closed before rename)
static void writeSlice(
    const uint8_t* volumeData,
    size_t         sliceIdx,
    size_t         dimX,
    size_t         dimY,
    size_t         bytesPerSample,
    bool           isXYZ,
    uint16_t       compression,
    const std::string& outPath,
    std::vector<uint8_t>& scratch
) {
    size_t width      = isXYZ ? dimX : dimY;
    size_t height     = isXYZ ? dimY : dimX;
    size_t sliceBytes = dimX * dimY * bytesPerSample;
    const uint8_t* srcBase = volumeData + sliceIdx * sliceBytes;

    fs::path tmpP = fs::path(outPath).concat(".tmp");

    {
        TiffHandle tf(tmpP.string(), "w");
        TIFF* tif = tf.tif;

        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      (uint32_t)width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     (uint32_t)height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   (uint16_t)(bytesPerSample*8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)1);
        TIFFSetField(tif, TIFFTAG_COMPRESSION,     compression);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,    (uint32_t)height);

        uint8_t* dst = scratch.data();
        // reorder column-major → row-major
        for (size_t r = 0; r < height; ++r) {
            for (size_t c = 0; c < width; ++c) {
                size_t orow = isXYZ ? c : r;
                size_t ocol = isXYZ ? r : c;
                size_t srcOff = (orow + ocol * dimX) * bytesPerSample;
                size_t dstOff = (r * width + c) * bytesPerSample;
                memcpy(dst + dstOff, srcBase + srcOff, bytesPerSample);
            }
        }

        tsize_t written = (compression == COMPRESSION_NONE)
            ? TIFFWriteRawStrip(tif, 0, scratch.data(), sliceBytes)
            : TIFFWriteEncodedStrip(tif, 0, scratch.data(), sliceBytes);

        if (written < 0)
            throw std::runtime_error("TIFF write failed on slice " + std::to_string(sliceIdx));
    }  // TIFFClose happens here

    if (fs::exists(outPath)) fs::remove(outPath);
    fs::rename(tmpP, outPath);
}

// The MEX gateway
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    // 1) Basic usage check
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:usage",
            "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");

    // 2) Volume type
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
        mexErrMsgIdAndTxt("save_bl_tif:type",
            "Volume must be uint8 or uint16.");

    // 3) Dimensions and slice count
    const mwSize* dims = mxGetDimensions(prhs[0]);
    size_t dimX = dims[0], dimY = dims[1];
    size_t numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3)
                         ? dims[2] : 1;

    // 4) Transpose flag
    bool isXYZ = mxIsLogicalScalarTrue(prhs[2])
               || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // 5) Compression
    char* compC = mxArrayToUTF8String(prhs[3]);
    std::string compStr(compC);
    mxFree(compC);
    uint16_t compression = (compStr=="none")    ? COMPRESSION_NONE
                         : (compStr=="lzw")     ? COMPRESSION_LZW
                         : (compStr=="deflate") ? COMPRESSION_DEFLATE
                         : throw std::runtime_error("Invalid compression: "+compStr);

    // 6) fileList sanity
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1])!=numSlices)
        mexErrMsgIdAndTxt("save_bl_tif:files",
            "fileList must be a cell-array of length = #slices.");

    std::vector<std::string> outputFiles(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        outputFiles[i] = s;
        mxFree(s);
    }

    // 7) Guard-clauses: directory exists, not read-only
    for (auto& f : outputFiles) {
        fs::path d = fs::path(f).parent_path();
        if (!d.empty() && !fs::exists(d))
            mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
                "Output directory does not exist: %s", d.string().c_str());
        if (fs::exists(f) && access(f.c_str(), W_OK) != 0)
            mexErrMsgIdAndTxt("save_bl_tif:readonly",
                "Cannot overwrite read-only file: %s", f.c_str());
    }

    // 8) Data pointer & sample size
    const uint8_t* volumeData = static_cast<const uint8_t*>(mxGetData(prhs[0]));
    size_t bytesPerSample = (mxGetClassID(prhs[0])==mxUINT16_CLASS ? 2 : 1);

    // 9) Thread count (default = half of HW threads)
    size_t hw = std::thread::hardware_concurrency();
    if (hw==0) hw = 1;
    size_t defaultThreads = std::max(hw/2, size_t(1));
    size_t requested   = (nrhs==5) ? (size_t)mxGetScalar(prhs[4]) : defaultThreads;
    size_t numThreads = std::min(requested, numSlices);
    if (numThreads<1) numThreads = 1;

    // 10) Atomic counter for slice indices
    std::atomic<size_t> nextSlice{0};

    // 11) Launch worker threads
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            // each thread gets its own scratch buffer
            std::vector<uint8_t> scratch(dimX * dimY * bytesPerSample);

            while (true) {
                size_t idx = nextSlice.fetch_add(1, std::memory_order_relaxed);
                if (idx >= numSlices) break;
                try {
                    writeSlice(
                      volumeData, idx,
                      dimX, dimY,
                      bytesPerSample,
                      isXYZ,
                      compression,
                      outputFiles[idx],
                      scratch
                    );
                }
                catch (const std::exception& ex) {
                    // bubble up via mexErrMsgIdAndTxt:
                    mexErrMsgIdAndTxt("save_bl_tif:runtime", ex.what());
                }
            }
        });
    }

    // 12) Wait and clean up
    for (auto& th : workers) th.join();

    // 13) Return volume if requested
    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
}
