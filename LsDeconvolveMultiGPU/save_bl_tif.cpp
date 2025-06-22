/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------

  High-throughput TIFF Z-slice saver for 3D MATLAB volumes.
  Each slice is saved to a separate TIFF file using a reusable thread pool.

  USAGE:
      save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])

  INPUT:
    volume      : 3D uint8/uint16 MATLAB array in [X Y Z] or [Y X Z] layout.
    fileList    : 1×Z cell array of filenames (one per Z-slice).
    isXYZ       : boolean or numeric, true if data is [X Y Z].
    compression : string, one of: "none", "lzw", or "deflate".
    nThreads    : (optional) number of threads (default = physical cores or Z count).

  FEATURES:
    • Fixed‐size thread pool with atomic slice dispatch
    • Per-thread scratch buffer reserved (no zero-initialization)
    • RAII for TIFF handles and clean error collection
    • Atomic replace (remove+rename) of temp files
    • Cross-platform checks with <unistd.h> for access()

  LIMITATIONS:
    • Grayscale only (1 channel)
    • No retry on I/O errors
    • Assumes local filesystem

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
#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>
#include <unistd.h>   // for access()

// RAII TIFF handle
struct TiffHandle {
    TIFF* tif;
    TiffHandle(const std::string& path, const char* mode)
      : tif(TIFFOpen(path.c_str(), mode)) {
        if (!tif) throw std::runtime_error("Cannot open TIFF: " + path);
    }
    ~TiffHandle() { if (tif) TIFFClose(tif); }
};

// Check file can be overwritten
inline void ensureWritable(const std::string& path) {
    if (access(path.c_str(), F_OK) == 0 &&
        access(path.c_str(), W_OK) != 0)
    {
        throw std::runtime_error("Cannot overwrite read-only file: " + path);
    }
}

// Determine physical core count on Linux
size_t countPhysicalCores() {
#ifdef __linux__
    std::vector<std::pair<int,int>> coreIds;
    FILE* info = std::fopen("/proc/cpuinfo", "r");
    if (!info) return std::thread::hardware_concurrency();
    int phys=-1, core=-1;
    char line[256];
    while (std::fgets(line, sizeof(line), info)) {
        if (std::sscanf(line, "physical id\t: %d", &phys)==1) continue;
        if (std::sscanf(line, "core id\t: %d", &core)==1)
            coreIds.emplace_back(phys, core);
    }
    std::fclose(info);
    std::sort(coreIds.begin(), coreIds.end());
    coreIds.erase(std::unique(coreIds.begin(), coreIds.end()), coreIds.end());
    return coreIds.size();
#else
    return std::thread::hardware_concurrency();
#endif
}

// Save one slice (may transpose if needed)
void saveSlice(
    const uint8_t* baseData,
    size_t byteOffset,
    mwSize rows, mwSize cols,
    bool isXYZ,
    uint16_t compressionTag,
    size_t bytesPerPixel,
    size_t sliceBytes,
    size_t sliceIndex,
    const std::string& outFilename,
    std::vector<uint8_t>& scratchBuffer
) {
    ensureWritable(outFilename);
    const uint8_t* slicePtr = baseData + byteOffset;
    mwSize outRows = isXYZ ? cols : rows;
    mwSize outCols = isXYZ ? rows : cols;

    // Open temp and set TIFF tags
    std::string tmpPath = outFilename + ".tmp";
    TiffHandle handle(tmpPath, "w");
    TIFF* tif = handle.tif;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  outCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, outRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,
                 (bytesPerPixel == 2) ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, outRows);

    const uint8_t* writeBuffer = nullptr;

    if (isXYZ && compressionTag == COMPRESSION_NONE) {
        // Straight write
        writeBuffer = slicePtr;
    } else {
        // Ensure capacity (does NOT zero)
        scratchBuffer.reserve(sliceBytes);
        uint8_t* buf = scratchBuffer.data();
        if (!isXYZ) {
            // Transpose [Y X Z] → [X Y]
            for (mwSize c = 0; c < outCols; ++c) {
                const uint8_t* colPtr = slicePtr + c * rows * bytesPerPixel;
                for (mwSize r = 0; r < outRows; ++r) {
                    size_t idx = (r * outCols + c) * bytesPerPixel;
                    std::memcpy(buf + idx, colPtr + r * bytesPerPixel, bytesPerPixel);
                }
            }
        } else {
            // Same layout, just copy
            std::memcpy(buf, slicePtr, sliceBytes);
        }
        writeBuffer = buf;
    }

    // Write strip (raw for none, encoded otherwise)
    tsize_t written = (compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif, 0, const_cast<uint8_t*>(writeBuffer), sliceBytes)
        : TIFFWriteEncodedStrip(tif, 0, const_cast<uint8_t*>(writeBuffer), sliceBytes);
    if (written < 0)
        throw std::runtime_error("TIFF write failed on slice " + std::to_string(sliceIndex));

    // Atomically replace
    std::remove(outFilename.c_str());
    if (std::rename(tmpPath.c_str(), outFilename.c_str()) != 0)
        throw std::runtime_error("Failed to rename slice " + std::to_string(sliceIndex));
}

// MEX entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    // Validate inputs
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "Usage: save_bl_tif(vol, fileList, isXYZ, comp[, nThreads])");

    const mxArray* volumeMx = prhs[0];
    if (!mxIsUint8(volumeMx) && !mxIsUint16(volumeMx))
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "Volume must be uint8 or uint16");

    // Get dimensions
    const mwSize* dims = mxGetDimensions(volumeMx);
    mwSize rows    = dims[0];
    mwSize cols    = dims[1];
    mwSize zSlices = (mxGetNumberOfDimensions(volumeMx) == 3 ? dims[2] : 1);

    bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                 (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    // Compression argument
    char* compChars = mxArrayToUTF8String(prhs[3]);
    std::string compStr(compChars);
    mxFree(compChars);
    uint16_t compressionTag =
         (compStr == "lzw")     ? COMPRESSION_LZW
       : (compStr == "deflate") ? COMPRESSION_DEFLATE
       : (compStr == "none")    ? COMPRESSION_NONE
       : throw std::runtime_error("Invalid compression type");

    // File list
    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != zSlices)
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "fileList must have one entry per Z slice");
    std::vector<std::string> outputFilenames(zSlices);
    for (mwSize i = 0; i < zSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        outputFilenames[i] = s;
        mxFree(s);
    }

    // Data pointers and sizes
    const uint8_t* baseData = static_cast<const uint8_t*>(mxGetData(volumeMx));
    size_t bytesPerPixel = (mxGetClassID(volumeMx) == mxUINT16_CLASS ? 2 : 1);
    size_t sliceBytes   = (size_t)rows * cols * bytesPerPixel;

    // Thread count
    size_t maxRequested = (nrhs == 5)
        ? static_cast<size_t>(mxGetScalar(prhs[4]))
        : countPhysicalCores();
    size_t threadCount = std::min(maxRequested, (size_t)zSlices);
    if (threadCount < 1) threadCount = 1;

    // Prepare for threaded dispatch
    std::atomic<size_t> nextSlice(0);
    std::vector<std::thread> workers;
    std::mutex errorMutex;
    std::vector<std::string> errorMessages;

    // Launch worker threads
    for (size_t t = 0; t < threadCount; ++t) {
        workers.emplace_back([&](){
            // Per-thread scratch (reserve only)
            std::vector<uint8_t> scratch;
            scratch.reserve(sliceBytes);

            while (true) {
                size_t idx = nextSlice.fetch_add(1, std::memory_order_relaxed);
                if (idx >= zSlices) break;
                try {
                    saveSlice(baseData,
                              idx * sliceBytes,
                              rows, cols,
                              isXYZ,
                              compressionTag,
                              bytesPerPixel,
                              sliceBytes,
                              idx,
                              outputFilenames[idx],
                              scratch);
                }
                catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lock(errorMutex);
                    errorMessages.push_back(ex.what());
                }
            }
        });
    }

    // Wait for all to finish
    for (auto& th : workers) th.join();

    // If any errors, report
    if (!errorMessages.empty()) {
        std::ostringstream oss;
        oss << "Errors occurred:\n";
        for (auto& e : errorMessages) oss << " - " << e << "\n";
        mexErrMsgIdAndTxt("save_bl_tif:Runtime", oss.str().c_str());
    }

    // Return the input volume if requested
    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(volumeMx);
    }
}
