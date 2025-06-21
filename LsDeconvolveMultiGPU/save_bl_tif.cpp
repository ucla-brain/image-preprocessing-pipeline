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

/*==============================================================================
  save_bl_tif.cpp (OpenMP + Latency-Minimized Version)
  -----------------------------------------------------------------------------
  Fast Z-slice TIFF saver with minimal MATLAB overhead and optimized dispatch.

  Author        : Keivan Moradi (with ChatGPT-4o assistance)
  License       : GNU GPL v3
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

struct SaveTask {
    const uint8_t* base;
    size_t offset_bytes;
    mwSize dim0, dim1;
    std::string path;
    bool isXYZ;
    mxClassID classId;
    uint16_t compressionTag;
    size_t bytesPerSlice;
    size_t bytesPerPixel;
};

static void save_slice(const SaveTask& t, std::vector<uint8_t>& scratch)
{
    const mwSize srcRows = t.isXYZ ? t.dim1 : t.dim0;
    const mwSize srcCols = t.isXYZ ? t.dim0 : t.dim1;
    const uint8_t* inputSlice = t.base + t.offset_bytes;
    const bool directWrite = (t.compressionTag == COMPRESSION_NONE && t.isXYZ);

    if (!directWrite) {
        if (scratch.size() < t.bytesPerSlice)
            scratch.resize(t.bytesPerSlice);
        uint8_t* dstBuffer = scratch.data();

        if (!t.isXYZ) {
            for (mwSize col = 0; col < srcCols; ++col) {
                const uint8_t* srcColumn = inputSlice + col * t.dim0 * t.bytesPerPixel;
                for (mwSize row = 0; row < srcRows; ++row) {
                    size_t dstIdx = (size_t(row) * srcCols + col) * t.bytesPerPixel;
                    const uint8_t* src = srcColumn + row * t.bytesPerPixel;
                    if (t.bytesPerPixel == 1)
                        dstBuffer[dstIdx] = *src;
                    else
                        std::memcpy(dstBuffer + dstIdx, src, 2);
                }
            }
        } else {
            const size_t rowBytes = srcCols * t.bytesPerPixel;
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
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, t.bytesPerPixel == 2 ? 16 : 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, t.compressionTag);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, srcRows);

    const tsize_t wrote = (t.compressionTag == COMPRESSION_NONE)
        ? TIFFWriteRawStrip(tif, 0, directWrite ? const_cast<uint8_t*>(inputSlice) : scratch.data(), t.bytesPerSlice)
        : TIFFWriteEncodedStrip(tif, 0, scratch.data(), t.bytesPerSlice);

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

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs != 4)
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                "Usage: save_bl_tif(vol, fileList, orderFlag, compression)");

        const mxArray* V = prhs[0];
        if (!mxIsUint8(V) && !mxIsUint16(V))
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Volume must be uint8/uint16");

        const mwSize* dims = mxGetDimensions(V);
        const size_t dim0 = dims[0];
        const size_t dim1 = dims[1];
        const size_t dim2 = (mxGetNumberOfDimensions(V) == 3) ? dims[2] : 1;

        const uint8_t* base = static_cast<const uint8_t*>(mxGetData(V));
        const mxClassID classId = mxGetClassID(V);
        const size_t bytesPerPixel = (classId == mxUINT16_CLASS) ? 2 : 1;
        const size_t bytesPerSlice = dim0 * dim1 * bytesPerPixel;

        const bool isXYZ =
            mxIsLogicalScalarTrue(prhs[2]) ||
            ((mxIsUint32(prhs[2]) || mxIsInt32(prhs[2])) &&
             *static_cast<uint32_t*>(mxGetData(prhs[2])));

        // --- Parse compression ONCE ---
        char* comp_cstr = mxArrayToUTF8String(prhs[3]);
        if (!comp_cstr)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "Failed to convert compression input");
        std::string comp(comp_cstr);
        mxFree(comp_cstr);
        uint16_t compressionTag = COMPRESSION_NONE;
        if (comp == "lzw")      compressionTag = COMPRESSION_LZW;
        else if (comp == "deflate") compressionTag = COMPRESSION_DEFLATE;
        else if (comp != "none")
            mexErrMsgIdAndTxt("save_bl_tif:Input",
                              "compression must be \"none\", \"lzw\", or \"deflate\"");

        // --- Extract all filenames at once ---
        if (!mxIsCell(prhs[1]) || mxGetNumberOfElements(prhs[1]) != dim2)
            mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList size mismatch");
        mxArray** file_cells = static_cast<mxArray**>(mxGetData(prhs[1]));

        std::vector<std::string> file_paths;
        file_paths.reserve(dim2);
        for (size_t i = 0; i < dim2; ++i) {
            if (!mxIsChar(file_cells[i]))
                mexErrMsgIdAndTxt("save_bl_tif:Input", "fileList must be cellstr");
            char* s = mxArrayToUTF8String(file_cells[i]);
            file_paths.emplace_back(s);
            mxFree(s);
        }

        // --- Prepare tasks ---
        std::vector<SaveTask> tasks;
        tasks.reserve(dim2);
        for (size_t z = 0; z < dim2; ++z) {
            tasks.push_back(SaveTask{
                base,
                z * bytesPerSlice,
                dim0,
                dim1,
                file_paths[z],
                isXYZ,
                classId,
                compressionTag,
                bytesPerSlice,
                bytesPerPixel
            });
        }

        std::vector<std::string> errors;

#pragma omp parallel
        {
            std::vector<uint8_t> scratch;
#pragma omp for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
                try {
                    save_slice(tasks[i], scratch);
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

        if (nlhs > 0)
            plhs[0] = const_cast<mxArray*>(prhs[0]);  // return original array
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("save_bl_tif:Exception", "%s", e.what());
    }
}
