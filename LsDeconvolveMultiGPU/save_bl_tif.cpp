/*==============================================================================
  save_bl_tif.cpp
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (1 TIFF per Z-slice)

  Author:       Keivan Moradi (in collaboration with ChatGPT-4o)
  License:      GNU General Public License v3.0 (https://www.gnu.org/licenses/)

  OVERVIEW
  -------
  • Purpose:
      Efficiently saves each Z-slice from a 3D MATLAB array to a separate TIFF
      file using LZW, Deflate, or no compression. Supports [X Y Z] or [Y X Z] input.

  • Highlights:
      – Accepts `uint8` or `uint16` MATLAB input arrays.
      – Fully cross-platform (uses libtiff).
      – Supports multithreading.
      – Compression: none, lzw, or deflate.
      – Matches `load_bl_tif.cpp` slice order and dimensions.

  PARALLELISM
  -----------
  • Parallel I/O is implemented using atomic index dispatching:
      Each worker thread atomically claims the next available task index
      using `std::atomic<size_t>::fetch_add`, avoiding locks or queues.
      This model scales efficiently for uniform workloads like per-slice saves.

  USAGE
  -----
      save_bl_tif(array3d, fileList, orderFlag, compression)

      • array3d      : 3D numeric array, uint8 or uint16
      • fileList     : 1xZ cell array of full path strings
      • orderFlag    : (logical or uint32 scalar)
                         true  = [X Y Z] input
                         false = [Y X Z] input (MATLAB default)
      • compression  : string: "none", "lzw", or "deflate"

  ==============================================================================*/

#include "mex.h"
#include "matrix.h"
#include "tiffio.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// RAII char converter
// -----------------------------------------------------------------------------
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr) mexErrMsgIdAndTxt("save_bl_tif:BadString",
                                    "Failed to convert string from mxArray");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
    MatlabString(const MatlabString&) = delete;
    MatlabString& operator=(const MatlabString&) = delete;
};

// -----------------------------------------------------------------------------
// Task descriptor (no big buffers)
// -----------------------------------------------------------------------------
struct SaveTask {
    const uint8_t* basePtr;   // pointer to full MATLAB array
    mwSize dim0;              // rows (Y)
    mwSize dim1;              // cols (X)
    mwSize z;                 // slice index
    std::string path;
    bool isXYZ;               // true → input was [X Y Z]
    mxClassID classId;
    std::string compression;
};

// -----------------------------------------------------------------------------
// Write one slice
// -----------------------------------------------------------------------------
void save_slice(const SaveTask& t) {
    const size_t elemSize = (t.classId == mxUINT16_CLASS) ? 2 : 1;
    const mwSize width   = t.dim1;  // X
    const mwSize height  = t.dim0;  // Y

    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,  width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (elemSize == 2 ? 16 : 8));

    uint16_t comp = COMPRESSION_NONE;
    if (t.compression == "lzw")      comp = COMPRESSION_LZW;
    else if (t.compression == "deflate") comp = COMPRESSION_DEFLATE;
    TIFFSetField(tif, TIFFTAG_COMPRESSION, comp);

    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);  // one strip

    const size_t sliceOffset = static_cast<size_t>(t.z) * t.dim0 * t.dim1;
    std::vector<uint8_t> scan(width * elemSize);      // small buffer

    for (mwSize y = 0; y < height; ++y) {
        for (mwSize x = 0; x < width; ++x) {
            mwSize srcIdx = isXYZ
                ?  y + x * dim0 + sliceOffset    // permuted [X Y Z] in memory
                :  y + x * dim0 + sliceOffset;   // native [Y X Z]
            std::memcpy(&scan[(x * elemSize)],
                        basePtr + srcIdx * elemSize, elemSize);
        }
        if (TIFFWriteScanline(tif, scan.data(), y, 0) < 0) {
            TIFFClose(tif);
            throw std::runtime_error("TIFFWriteScanline failed on " + t.path);
        }
    }
    TIFFClose(tif);
}

// -----------------------------------------------------------------------------
// Worker thread
// -----------------------------------------------------------------------------
void worker(const std::vector<SaveTask>& tasks,
            std::atomic_size_t& next,
            std::vector<std::string>& errors,
            std::mutex& err_mtx) {
    size_t i;
    while ((i = next.fetch_add(1)) < tasks.size()) {
        try { save_slice(tasks[i]); }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(err_mtx);
            errors.emplace_back(e.what());
        }
    }
}

// -----------------------------------------------------------------------------
// mexFunction
// -----------------------------------------------------------------------------
void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Usage: save_bl_tif(vol3d, fileList, orderFlag, compression)");

        const mxArray* vol = prhs[0];
        if (!mxIsUint8(vol) && !mxIsUint16(vol))
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Input must be uint8 or uint16");

        if (mxGetNumberOfDimensions(vol) != 3)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "Input must be 3-D");

        mwSize dim0 = mxGetDimensions(vol)[0];   // rows(Y)
        mwSize dim1 = mxGetDimensions(vol)[1];   // cols(X)
        mwSize dim2 = mxGetDimensions(vol)[2];   // slices(Z)

        if (!mxIsCell(prhs[1]) ||
            mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "fileList must be 1xZ cell array");

        bool isXYZ = false;
        const mxArray* flag = prhs[2];
        if (mxIsLogicalScalar(flag)) isXYZ = mxIsLogicalScalarTrue(flag);
        else if (mxIsUint32(flag) || mxIsInt32(flag))
            isXYZ = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "orderFlag must be logical or int32/uint32 scalar");

        MatlabString compStr(prhs[3]);
        std::string compression(compStr.get());
        if (compression != "none" && compression != "lzw" && compression != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be 'none', 'lzw', or 'deflate'");

        // Build tasks
        const uint8_t* basePtr = static_cast<const uint8_t*>(mxGetData(vol));
        std::vector<SaveTask> tasks;
        tasks.reserve(dim2);

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* cell = mxGetCell(prhs[1], z);
            if (!mxIsChar(cell))
                mexErrMsgIdAndTxt("save_bl_tif:Input",
                                  "fileList entries must be char");
            MatlabString p(cell);
            tasks.push_back({basePtr, dim0, dim1, z, p.get(), isXYZ,
                             mxGetClassID(vol), compression});
        }

        // Multithread
        std::atomic_size_t next(0);
        std::vector<std::string> errors;
        std::mutex err_mtx;
        unsigned nThreads = std::max(1u,
            std::min<unsigned>(std::thread::hardware_concurrency(), dim2));
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nThreads; ++t)
            pool.emplace_back(worker, std::cref(tasks),
                              std::ref(next), std::ref(errors),
                              std::ref(err_mtx));
        for (auto& th : pool) th.join();

        if (!errors.empty()) {
            std::string msg = "save_bl_tif errors:\n";
            for (auto& e : errors) msg += "  - " + e + "\n";
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}