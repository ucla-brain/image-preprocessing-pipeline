/*==============================================================================
  save_bl_tif.cpp
  ------------------------------------------------------------------------------

  High-throughput TIFF Z-slice saver for 3D MATLAB volumes.
  Each slice is saved to a separate TIFF file using OpenMP for multithreading.

  USAGE:
      save_bl_tif(volume, fileList, isXYZ, compression[, nThreads])

  INPUT:
    volume      : 3D uint8/uint16 MATLAB array in [X Y Z] or [Y X Z] layout.
    fileList    : 1×Z cell array of filenames (one per Z-slice).
    isXYZ       : boolean or numeric, true if data is [X Y Z].
    compression : string, one of: "none", "lzw", or "deflate".
    nThreads    : (optional) number of threads (default = physical cores or Z count).

  FEATURES:
    • Uses OpenMP parallel for with dynamic scheduling and num_threads clause.
    • Thread-local scratch buffer for efficient transpose/copy.
    • RAII for TIFF file handles.
    • Atomic file replace via std::remove + std::rename.
    • Cross-platform I/O with <unistd.h> for access().
    • Errors collected safely and reported to MATLAB.

  LIMITATIONS:
    • Grayscale only (1 channel).
    • No retry on I/O errors.
    • Assumes local filesystem.

  AUTHOR:
    Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE:
    GNU GPL v3 — https://www.gnu.org/licenses/gpl-3.0.html
==============================================================================*/

#include "mex.h"
#include "tiffio.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <mutex>
#include <thread>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include <unistd.h>   // for access()

// Thread-local buffer for transpose/compressed writes
struct ScratchBuffer {
    std::vector<uint8_t> buf;
    void ensure(size_t bytes) {
        if (buf.size() < bytes) buf.resize(bytes);
    }
};
static thread_local ScratchBuffer threadScratch;

// RAII wrapper for TIFF*
struct TiffHandle {
    TIFF* tif;
    TiffHandle(const std::string& path, const char* mode)
      : tif(TIFFOpen(path.c_str(), mode)) {
        if (!tif) throw std::runtime_error("Cannot open TIFF: " + path);
    }
    ~TiffHandle() { if (tif) TIFFClose(tif); }
};

// Ensure we can overwrite an existing file
inline void ensureWritable(const std::string& path) {
    if (access(path.c_str(), F_OK) == 0 &&
        access(path.c_str(), W_OK) != 0)
    {
        throw std::runtime_error("Cannot overwrite read-only file: " + path);
    }
}

// Save one Z-slice to disk
void saveSlice(const uint8_t* basePtr, size_t byteOffset,
               mwSize numRows, mwSize numCols, bool isXYZ,
               uint16_t compTag, size_t bytesPerPixel,
               size_t bytesPerSlice, size_t sliceIndex,
               const std::string& filename)
{
    ensureWritable(filename);
    const uint8_t* sliceData = basePtr + byteOffset;
    mwSize outRows = isXYZ ? numCols : numRows;
    mwSize outCols = isXYZ ? numRows : numCols;

    std::string tmpPath = filename + ".tmp";
    TiffHandle handle(tmpPath, "w");
    TIFF* tif = handle.tif;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  outCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, outRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,
                 (bytesPerPixel == 2) ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, outRows);

    const uint8_t* writeBuf = nullptr;
    if (isXYZ && compTag == COMPRESSION_NONE) {
        writeBuf = sliceData;
    } else {
        threadScratch.ensure(bytesPerSlice);
        uint8_t* buf = threadScratch.buf.data();
        if (!isXYZ) {
            // transpose [Y X Z] → [X Y]
            for (mwSize c = 0; c < outCols; ++c) {
                const uint8_t* colPtr = sliceData + c * numRows * bytesPerPixel;
                for (mwSize r = 0; r < outRows; ++r) {
                    size_t idx = (r * outCols + c) * bytesPerPixel;
                    std::memcpy(buf + idx, colPtr + r * bytesPerPixel, bytesPerPixel);
                }
            }
        } else {
            std::memcpy(buf, sliceData, bytesPerSlice);
        }
        writeBuf = buf;
    }

    tsize_t written = (compTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif, 0, const_cast<uint8_t*>(writeBuf), bytesPerSlice)
        : TIFFWriteEncodedStrip(tif, 0, const_cast<uint8_t*>(writeBuf), bytesPerSlice);
    if (written < 0)
        throw std::runtime_error("TIFF write failed on slice " + std::to_string(sliceIndex));

    // Atomic replace: remove old, rename new
    std::remove(filename.c_str());
    if (std::rename(tmpPath.c_str(), filename.c_str()) != 0)
        throw std::runtime_error("Failed to rename slice " + std::to_string(sliceIndex));
}

// Count physical cores on Linux, or fallback
size_t countPhysicalCores() {
#ifdef __linux__
    std::vector<std::pair<int,int>> ids;
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return std::thread::hardware_concurrency();
    int phys=-1, core=-1;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "physical id\t: %d", &phys)==1) continue;
        if (std::sscanf(line, "core id\t: %d", &core)==1)
            ids.emplace_back(phys, core);
    }
    std::fclose(f);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids.size();
#else
    return std::thread::hardware_concurrency();
#endif
}

// MATLAB MEX entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "Usage: save_bl_tif(vol, fileList, isXYZ, comp[, nThreads])");

    const mxArray* volumeMx = prhs[0];
    if (!mxIsUint8(volumeMx) && !mxIsUint16(volumeMx))
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "Volume must be uint8 or uint16");

    const mwSize* dims = mxGetDimensions(volumeMx);
    mwSize numRows = dims[0], numCols = dims[1];
    mwSize zSlices = (mxGetNumberOfDimensions(volumeMx) == 3 ? dims[2] : 1);
    bool isXYZ = mxIsLogicalScalarTrue(prhs[2]) ||
                 (mxIsNumeric(prhs[2]) && mxGetScalar(prhs[2]) != 0);

    char* compUTF = mxArrayToUTF8String(prhs[3]);
    std::string compStr(compUTF);
    mxFree(compUTF);
    uint16_t compTag = (compStr == "lzw")     ? COMPRESSION_LZW :
                       (compStr == "deflate") ? COMPRESSION_DEFLATE :
                       (compStr == "none")    ? COMPRESSION_NONE :
                       throw std::runtime_error("Invalid compression type");

    if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != zSlices)
        mexErrMsgIdAndTxt("save_bl_tif:Input",
            "fileList must have one entry per Z slice");
    std::vector<std::string> fileList(zSlices);
    for (mwSize i = 0; i < zSlices; ++i) {
        char* s = mxArrayToUTF8String(mxGetCell(prhs[1], i));
        fileList[i] = s;
        mxFree(s);
    }

    const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(volumeMx));
    size_t bytesPerPixel = (mxGetClassID(volumeMx) == mxUINT16_CLASS) ? 2 : 1;
    size_t bytesPerSlice = numRows * numCols * bytesPerPixel;

    // Determine number of threads
    size_t maxThreads = (nrhs == 5)
        ? static_cast<size_t>(mxGetScalar(prhs[4]))
        : countPhysicalCores();
    size_t numThreads = std::min(maxThreads, static_cast<size_t>(zSlices));
    if (numThreads < 1) numThreads = 1;

    std::vector<std::string> errors;
    std::mutex errorMutex;

    #pragma omp parallel for schedule(dynamic,1) num_threads(numThreads)
    for (mwSize idx = 0; idx < zSlices; ++idx) {
        try {
            saveSlice(basePtr, idx * bytesPerSlice,
                      numRows, numCols, isXYZ,
                      compTag, bytesPerPixel,
                      bytesPerSlice, idx,
                      fileList[idx]);
        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(errorMutex);
            errors.push_back(e.what());
        }
    }

    if (!errors.empty()) {
        std::ostringstream msg;
        msg << "Errors occurred:\n";
        for (auto& e : errors) msg << " - " << e << "\n";
        mexErrMsgIdAndTxt("save_bl_tif:Runtime", msg.str().c_str());
    }

    if (nlhs > 0) {
        plhs[0] = const_cast<mxArray*>(volumeMx);
    }
}
