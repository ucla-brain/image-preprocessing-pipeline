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
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <chrono>     // ← for std::chrono::seconds
#include <thread>     // ← for std::this_thread::sleep_for

struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("save_bl_tif:BadString", "Failed to convert string from mxArray");
    }
    MatlabString(const MatlabString&) = delete;
    MatlabString& operator=(const MatlabString&) = delete;
    MatlabString(MatlabString&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    MatlabString& operator=(MatlabString&& other) noexcept {
        if (this != &other) {
            mxFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    operator const char*() const { return ptr; }
};

struct SaveTask {
    std::vector<uint8_t> buffer;
    std::string path;
    mwSize width;
    mwSize height;
    mxClassID classId;
    std::string compression;
};

void save_slice(const SaveTask& task) {
    TIFF* tif = TIFFOpen(task.path.c_str(), "w");
    if (!tif)
        throw std::runtime_error("Failed to open file for writing: " + task.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, task.width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, task.height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (task.classId == mxUINT16_CLASS ? 16 : 8));
    TIFFSetField(tif, TIFFTAG_COMPRESSION,
                 (task.compression == "lzw" ? COMPRESSION_LZW :
                  task.compression == "deflate" ? COMPRESSION_DEFLATE : COMPRESSION_NONE));
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, task.height);

    size_t rowSize = task.width * (task.classId == mxUINT16_CLASS ? 2 : 1);
    for (mwSize row = 0; row < task.height; ++row) {
        const uint8_t* buf = task.buffer.data() + row * rowSize;
        if (TIFFWriteScanline(tif, (void*)buf, row, 0) < 0) {
            TIFFClose(tif);
            throw std::runtime_error("TIFFWriteScanline failed: " + task.path);
        }
    }
    TIFFClose(tif);
}

void worker_main(const std::vector<SaveTask>& tasks, std::atomic<size_t>& next,
                 std::vector<std::string>& errors, std::mutex& err_mutex) {
    size_t i;
    while ((i = next.fetch_add(1)) < tasks.size()) {
        const SaveTask& task = tasks[i];
        const int max_retries = 40;
        bool success = false;
        for (int attempt = 1; attempt <= max_retries; ++attempt) {
            try {
                save_slice(task);
                success = true;
                break;
            } catch (const std::exception& e) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        if (!success) {
            std::lock_guard<std::mutex> lock(err_mutex);
            errors.emplace_back("Exceeded max retries: " + task.path);
        }
    }
}

void mexFunction(int nlhs, mxArray*[], int nrhs, const mxArray* prhs[]) {
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Usage: save_bl_tif(array3d, fileList, orderFlag, compression)");

        const mxArray* array = prhs[0];
        if (!mxIsUint8(array) && !mxIsUint16(array))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Input array must be uint8 or uint16");

        const mwSize* dims = mxGetDimensions(array);
        if (mxGetNumberOfDimensions(array) != 3)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Input must be a 3D array");

        mwSize dim0 = dims[0], dim1 = dims[1], dim2 = dims[2];

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be a 1xZ cell array matching 3rd dim of input");

        bool isXYZ = false;
        const mxArray* flag = prhs[2];
        if (mxIsLogicalScalar(flag))
            isXYZ = mxIsLogicalScalarTrue(flag);
        else if ((mxIsInt32(flag) || mxIsUint32(flag)) && mxGetNumberOfElements(flag) == 1)
            isXYZ = (*static_cast<uint32_t*>(mxGetData(flag)) != 0);
        else
            mexErrMsgIdAndTxt("save_bl_tif:Input", "orderFlag must be logical or int32/uint32 scalar");

        MatlabString compStr(prhs[3]);
        std::string compression(compStr);
        if (compression != "none" && compression != "lzw" && compression != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input", "compression must be 'none', 'lzw', or 'deflate'");

        std::vector<SaveTask> tasks;
        tasks.reserve(dim2);

        mxClassID classId = mxGetClassID(array);
        size_t elemSize = (classId == mxUINT16_CLASS) ? 2 : 1;

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* cell = mxGetCell(prhs[1], z);
            if (!mxIsChar(cell))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "Each file path must be a string");
            MatlabString path(cell);

            mwSize width  = dim1;  // X dimension
            mwSize height = dim0; // Y dimension
            std::vector<uint8_t> buf(width * height * elemSize);

            const uint8_t* data = static_cast<const uint8_t*>(mxGetData(array));

            for (mwSize y = 0; y < height; ++y) {
                for (mwSize x = 0; x < width; ++x) {
                    size_t dst_idx = (y * width + x) * elemSize;
                    mwSize src_idx;
                    if (isXYZ)
                        src_idx = y + x * dim0 + z * dim0 * dim1;
                    else
                        src_idx = y + x * dim0 + z * dim0 * dim1;
                    std::memcpy(&buf[dst_idx], &data[src_idx * elemSize], elemSize);
                }
            }

            tasks.push_back({std::move(buf), path.get(), width, height, classId, compression});
        }

        std::atomic<size_t> next(0);
        std::vector<std::string> errors;
        std::mutex err_mutex;

        unsigned numThreads = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), dim2));
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < numThreads; ++t)
            threads.emplace_back(worker_main, std::cref(tasks), std::ref(next), std::ref(errors), std::ref(err_mutex));
        for (auto& th : threads) th.join();

        if (!errors.empty()) {
            std::ostringstream oss;
            oss << "save_bl_tif failed:\n";
            for (const auto& e : errors) oss << "  - " << e << "\n";
            mexErrMsgIdAndTxt("save_bl_tif:Error", "%s", oss.str().c_str());
        }
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
