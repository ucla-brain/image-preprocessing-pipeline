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
    • Multi-row strips (default 64 rows) for bounded RAM & better compression.
    • Proper close-before-rename (avoids lockups under NUMA).
    • C++17 filesystem, RAII for TIFF handles.
    • Exception aggregation and reporting (main thread only).

  LIMITATIONS:
    • Grayscale only (single channel).
    • Single-strip TIFF per row-block.

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
#include <algorithm>
#include <unistd.h>

namespace fs = std::filesystem;

// RAII wrapper to ensure TIFF* is closed promptly
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

// Write one slice using multi-row strips (default 64 rows per strip)
static void writeSlice(
    const uint8_t*        volumeData,
    size_t                sliceIdx,
    size_t                dimX,
    size_t                dimY,
    size_t                bytesPerSample,
    bool                  isXYZ,
    uint16_t              compression,
    const std::string&    outPath
) {
    const size_t width  = isXYZ ? dimX : dimY;
    const size_t height = isXYZ ? dimY : dimX;
    const size_t sliceBytes = dimX * dimY * bytesPerSample;
    const uint8_t* srcBase = volumeData + sliceIdx * sliceBytes;

    // tmp filename
    fs::path tmp = fs::path(outPath).concat(".tmp");

    {
        TiffHandle tf(tmp.string(), "w");
        TIFF* tif = tf.tif;

        // set tags
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,   (uint32_t)width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,  (uint32_t)height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,(uint16_t)(bytesPerSample*8));
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,(uint16_t)1);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

        // choose your strip height
        uint32_t rowsPerStrip = std::min<uint32_t>(height, 64u);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

        uint32_t nStrips = (height + rowsPerStrip - 1) / rowsPerStrip;

        // buffer just ONE strip at a time
        std::vector<uint8_t> stripBuf(width * rowsPerStrip * bytesPerSample);

        for (uint32_t s = 0; s < nStrips; ++s) {
            uint32_t row0       = s * rowsPerStrip;
            uint32_t actualRows = std::min<uint32_t>(rowsPerStrip, height - row0);
            size_t   byteCount  = width * actualRows * bytesPerSample;

            // scatter/gather only these actualRows
            for (uint32_t r = 0; r < actualRows; ++r) {
                for (size_t c = 0; c < width; ++c) {
                    size_t orow   = isXYZ ? c : (row0 + r);
                    size_t ocol   = isXYZ ? (row0 + r) : c;
                    size_t srcOff = (orow + ocol * dimX) * bytesPerSample;
                    size_t dstOff = (r * width + c) * bytesPerSample;
                    memcpy(&stripBuf[dstOff], srcBase + srcOff, bytesPerSample);
                }
            }

            tsize_t written = (compression == COMPRESSION_NONE)
              ? TIFFWriteRawStrip   (tif, s, stripBuf.data(), byteCount)
              : TIFFWriteEncodedStrip(tif, s, stripBuf.data(), byteCount);

            if (written < 0)
                throw std::runtime_error("TIFF write failed on strip " + std::to_string(s));
        }
    }
    // swap in atomically
    if (fs::exists(outPath)) fs::remove(outPath);
    fs::rename(tmp, outPath);
}

// MEX entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    // 1) Usage
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:usage",
          "Usage: save_bl_tif(vol, fileList, isXYZ, compression[, nThreads]);");

    // 2) Type check
    if (!mxIsUint8(prhs[0]) && !mxIsUint16(prhs[0]))
        mexErrMsgIdAndTxt("save_bl_tif:type",
          "Volume must be uint8 or uint16.");

    // 3) Dimensions
    const mwSize* dims = mxGetDimensions(prhs[0]);
    size_t dimX      = dims[0];
    size_t dimY      = dims[1];
    size_t numSlices = (mxGetNumberOfDimensions(prhs[0]) == 3)
                         ? dims[2] : 1;

    // 4) Transpose flag
    bool isXYZ = mxIsLogicalScalarTrue(prhs[2])
               || (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2])!=0);

    // 5) Compression mode
    char* compC = mxArrayToUTF8String(prhs[3]);
    std::string compStr(compC);
    mxFree(compC);

    uint16_t compression = (compStr=="none")    ? COMPRESSION_NONE
                         : (compStr=="lzw")     ? COMPRESSION_LZW
                         : (compStr=="deflate") ? COMPRESSION_DEFLATE
                         : throw std::runtime_error("Invalid compression: "+compStr);

    // 6) fileList
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1])!=numSlices)
        mexErrMsgIdAndTxt("save_bl_tif:files",
          "fileList must be a cell-array of length = #slices.");

    std::vector<std::string> outputFiles(numSlices);
    for (size_t i = 0; i < numSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        outputFiles[i] = s;
        mxFree(s);
    }

    // 7) Guard-clauses: directory exists & not read-only
    for (auto& f : outputFiles) {
        fs::path d = fs::path(f).parent_path();
        if (!d.empty() && !fs::exists(d))
            mexErrMsgIdAndTxt("save_bl_tif:invalidPath",
              "Output directory does not exist: %s", d.string().c_str());
        if (fs::exists(f) && access(f.c_str(), W_OK)!=0)
            mexErrMsgIdAndTxt("save_bl_tif:readonly",
              "Cannot overwrite read-only file: %s", f.c_str());
    }

    // 8) Data pointer & sample size
    const uint8_t* volumeData = static_cast<const uint8_t*>(mxGetData(prhs[0]));
    size_t bytesPerSample = (mxGetClassID(prhs[0])==mxUINT16_CLASS ? 2 : 1);

    // 9) Thread count (default = half hardware threads)
    size_t hw = std::thread::hardware_concurrency();
    if (!hw) hw = 1;
    size_t defaultThreads = std::max(hw/2, size_t(1));
    size_t req = (nrhs==5) ? (size_t)mxGetScalar(prhs[4]) : defaultThreads;
    size_t numThreads = std::min(req, numSlices);
    if (!numThreads) numThreads = 1;

    // 10) Error collection
    std::vector<std::string> threadErrors;
    std::mutex              errorLock;

    // 11) Atomic slice index
    std::atomic<size_t> nextSlice{0};

    // 12) Launch workers
    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    for (size_t t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            // each thread just loops over slices:
            while (true) {
              size_t idx = nextSlice.fetch_add(1, std::memory_order_relaxed);
              if (idx >= numSlices) break;
              try {
                writeSlice(volumeData, idx,
                           dimX, dimY, bytesPerSample,
                           isXYZ, compression,
                           outputFiles[idx]);
              } catch (const std::exception &ex) {
                std::lock_guard<std::mutex> lk(errorLock);
                threadErrors.push_back(ex.what());
                break;  // stop this thread
              }
            }
        });
    }

    // 13) Join
    for (auto& th : workers) th.join();

    // 14) Report first error (main thread only)
    if (!threadErrors.empty()) {
        mexErrMsgIdAndTxt("save_bl_tif:runtime", threadErrors.front().c_str());
    }

    // 15) Return volume if requested
    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(prhs[0]);
    }
}
