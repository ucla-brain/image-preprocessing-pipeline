/*==============================================================================
  save_bl_tif.cpp (OpenMP Version)
  -----------------------------------------------------------------------------
  High-throughput Z-slice saver for 3-D MATLAB arrays (one TIFF per slice).

  Author        : Keivan Moradi (with ChatGPT-4o assistance)
  License       : GNU GPL v3 <https://www.gnu.org/licenses/>

  OVERVIEW
  -------
  • Purpose
      Save each Z-slice of a 3-D array to its own TIFF file. Optimized for
      [Y X Z] (MATLAB default) and [X Y Z] order, with or without compression.

  • Parallelism
      Uses OpenMP with dynamic scheduling. Threads are reused by runtime.

  • Features
      – Supports uint8/uint16
      – Uses TIFFWriteRawStrip if compression="none" + [X Y Z]
      – Fully drop-in for MATLAB
      – Thread-local scratch buffer per OpenMP thread

==============================================================================*/

#include "mex.h"
#include "tiffio.h"
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>
#include <omp.h>

#if defined(__linux__)
# include <fcntl.h>
# include <unistd.h>
#endif

struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* a) : ptr(mxArrayToUTF8String(a)) {
        if (!ptr)
            mexErrMsgIdAndTxt("save_bl_tif:BadString",
                              "mxArrayToUTF8String returned null");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const { return ptr; }
    MatlabString(const MatlabString&)            = delete;
    MatlabString& operator=(const MatlabString&) = delete;
};

struct SaveTask {
    const uint8_t* base;
    mwSize dim0, dim1, z;
    std::string path;
    bool isXYZ;
    mxClassID classId;
    std::string comp;
};

static void save_slice(const SaveTask& t, std::vector<uint8_t>& scratch)
{
    const size_t bytesPerPixel = (t.classId == mxUINT16_CLASS ? 2 : 1);
    const mwSize srcRows = t.isXYZ ? t.dim1 : t.dim0;
    const mwSize srcCols = t.isXYZ ? t.dim0 : t.dim1;
    const size_t pixelsPerSlice = static_cast<size_t>(t.dim0) * t.dim1;
    const size_t bytesPerSlice = pixelsPerSlice * bytesPerPixel;
    const size_t sliceIndex = static_cast<size_t>(t.z);
    const bool isRaw = (t.comp == "none");

    const uint8_t* inputSlice = t.base + sliceIndex * bytesPerSlice;
    const bool directWrite = isRaw && t.isXYZ;

    if (!directWrite) {
        if (scratch.size() < bytesPerSlice)
            scratch.resize(bytesPerSlice);
        uint8_t* dstBuffer = scratch.data();

        if (!t.isXYZ) {
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn =
                    inputSlice + static_cast<size_t>(col) * t.dim0 * bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (static_cast<size_t>(row) * srcCols + col) * bytesPerPixel;
                    const uint8_t* src = srcColumn + static_cast<size_t>(row) * bytesPerPixel;
                    if (bytesPerPixel == 1)
                        dstBuffer[dstIdx] = *src;
                    else
                        std::memcpy(dstBuffer + dstIdx, src, 2);
                }
            }
        } else {
            const size_t rowBytes = srcCols * bytesPerPixel;
            for (mwSize row = 0; row < srcRows; ++row) {
                const uint8_t* src = inputSlice + row * rowBytes;
                std::memcpy(dstBuffer + row * rowBytes, src, rowBytes);
            }
        }
    }

    TIFF* tif = TIFFOpen(t.path.c_str(), "w");
    if (!tif) throw std::runtime_error("Cannot open " + t.path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, srcCols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, srcRows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bytesPerPixel == 2 ? 16 : 8);

    const uint16_t compTag = isRaw ? COMPRESSION_NONE :
        (t.comp == "lzw" ? COMPRESSION_LZW : COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    const tsize_t wrote = isRaw
        ? TIFFWriteRawStrip(tif, 0, directWrite ? const_cast<uint8_t*>(inputSlice)
                                                : scratch.data(), bytesPerSlice)
        : TIFFWriteEncodedStrip(tif, 0, scratch.data(), bytesPerSlice);

    if (wrote < 0) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF write failed on " + t.path);
    }

#if defined(__linux__)
    int fd = TIFFFileno(tif);
    if (fd != -1) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    TIFFClose(tif);
}

void mexFunction(int, mxArray*[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        const mwSize* dims = mxGetDimensions(V);
        mwSize nd = mxGetNumberOfDimensions(V);
        mwSize dim0, dim1, dim2;
        if (nd == 2) { dim0 = dims[0]; dim1 = dims[1]; dim2 = 1; }
        else if (nd == 3) { dim0 = dims[0]; dim1 = dims[1]; dim2 = dims[2]; }
        else mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be 2-D or 3-D.");

        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");

        const bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
             *static_cast<uint32_t*>(mxGetData(prhs[2])));

        MatlabString cs(prhs[3]);
        std::string comp(cs.get());
        if (comp != "none" && comp != "lzw" && comp != "deflate")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "compression must be \"none\", \"lzw\", or \"deflate\"");

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        std::vector<SaveTask> tasks;
        tasks.reserve(dim2);

        for (mwSize z = 0; z < dim2; ++z) {
            const mxArray* cell = mxGetCell(prhs[1], z);
            if (!mxIsChar(cell))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be char");
            MatlabString path(cell);
            tasks.push_back({ base, dim0, dim1, z, path.get(),
                              isXYZ, mxGetClassID(V), comp });
        }

        std::vector<std::string> errors;

#pragma omp parallel
        {
            std::vector<uint8_t> threadScratch;
#pragma omp for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
                try {
                    save_slice(tasks[i], threadScratch);
                } catch (const std::exception& e) {
#pragma omp critical
                    errors.emplace_back(e.what());
                }
            }
        }

        if (!errors.empty()) {
            std::string msg("save_bl_tif errors:\n");
            for (const auto& e : errors) msg += "  - " + e + '\n';
            mexErrMsgIdAndTxt("save_bl_tif:Runtime", "%s", msg.c_str());
        }
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
