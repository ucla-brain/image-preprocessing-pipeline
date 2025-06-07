// ============================================================================
//  load_bl_tif.cpp  – Fast sub-region TIFF loader for MATLAB
//
//  Patched 2025-06-07
//     • tile & strip caching
//     • TIFFSwabArrayOfShort use
//     • 64-bit-safe indexing
//     • thread-pool fixes & misc. clean-ups
//     • error messages improved, robustness checks added
// ============================================================================

#include "mex.h"
#include "tiffio.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
//  Config & helpers
// ---------------------------------------------------------------------------
constexpr uint16_t kSupportedBitDepth8  = 8;
constexpr uint16_t kSupportedBitDepth16 = 16;

struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString",
                              "Failed to convert string from mxArray");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get() const noexcept { return ptr; }
    operator const char*() const noexcept { return ptr; }
};

struct LoadTask {
    int  in_row0, in_col0;      // ROI origin inside TIFF
    int  out_row0, out_col0;    // output start in output array
    int  cropH,  cropW;         // ROI size actually inside image
    int  roiH,   roiW;          // full requested ROI (for dst strides)
    int  zIndex;                // slice index
    size_t pixelsPerSlice;      // dst slice stride (roiH*roiW)
    std::string path;
    bool transpose;
};

struct TiffCloser {
    void operator()(TIFF* tif) const noexcept { if (tif) TIFFClose(tif); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// dst index helper (64-bit safe)
inline size_t computeDstIndex(const LoadTask& t, int r, int c) noexcept {
    if (!t.transpose) {
        return static_cast<size_t>(t.out_row0 + r)
             + static_cast<size_t>(t.out_col0 + c) * t.roiH
             + static_cast<size_t>(t.zIndex) * t.pixelsPerSlice;
    }
    return static_cast<size_t>(t.out_col0 + c)
         + static_cast<size_t>(t.out_row0 + r) * t.roiW
         + static_cast<size_t>(t.zIndex) * t.pixelsPerSlice;
}

// ---------------------------------------------------------------------------
//  Core read helpers
// ---------------------------------------------------------------------------

// Swap a whole uint16_t buffer in-place (vectorised in libtiff)
inline void swap_uint16_buf(void* buf, size_t count) noexcept {
    TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(buf),
                         static_cast<tmsize_t>(count));
}

// Read/copy a cropped sub-region from TIFF into caller-supplied buffer
static void readSubRegionToBuffer(const LoadTask& task,
                                  TIFF*           tif,
                                  uint8_t         bytesPerPixel,
                                  uint8_t*        dst /* cropH*cropW*BPP */)
{
    uint32_t imgW = 0, imgH = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);

    const bool needSwap16 =
        (bytesPerPixel == 2) && TIFFIsByteSwapped(tif);

    // --- Tiled ----------------------------------------------------------------
    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (!tileW || !tileH)
            throw std::runtime_error("Invalid tile size in TIFF metadata");

        const tmsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0)
            throw std::runtime_error("Invalid TIFFTileSize()");

        std::vector<uint8_t> tilebuf(static_cast<size_t>(tileSize));
        uint32_t prevTile = UINT32_MAX;

        for (int r = 0; r < task.cropH; ++r) {
            uint32_t imgY = static_cast<uint32_t>(task.in_row0 + r);
            for (int c = 0; c < task.cropW; ++c) {
                uint32_t imgX = static_cast<uint32_t>(task.in_col0 + c);

                const uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (tileIdx != prevTile) {
                    if (TIFFReadEncodedTile(tif, tileIdx,
                                            tilebuf.data(), tileSize) < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed");
                    if (needSwap16)
                        swap_uint16_buf(tilebuf.data(),
                                        static_cast<size_t>(tileSize / 2));
                    prevTile = tileIdx;
                }

                const uint32_t relY = imgY % tileH;
                const uint32_t relX = imgX % tileW;
                const size_t   srcOff =
                    (relY * tileW + relX) * bytesPerPixel;
                const size_t   dstOff =
                    (static_cast<size_t>(r) * task.cropW + c) * bytesPerPixel;

                std::memcpy(dst + dstOff, tilebuf.data() + srcOff, bytesPerPixel);
            }
        }
        return;
    }

    // --- Strips ---------------------------------------------------------------
    uint32_t rowsPerStrip = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
    if (!rowsPerStrip) rowsPerStrip = imgH;

    const size_t stripBytes =
        static_cast<size_t>(rowsPerStrip) * imgW * bytesPerPixel;
    if (stripBytes == 0 || stripBytes > (1u << 31))
        throw std::runtime_error("Invalid strip buffer size");

    std::vector<uint8_t> stripbuf(stripBytes);
    tstrip_t prevStrip = static_cast<tstrip_t>(-1);

    for (int r = 0; r < task.cropH; ++r) {
        const uint32_t imgY = static_cast<uint32_t>(task.in_row0 + r);
        const tstrip_t stripIdx = TIFFComputeStrip(tif, imgY, 0);

        if (stripIdx != prevStrip) {
            const tmsize_t n =
                TIFFReadEncodedStrip(tif, stripIdx,
                                     stripbuf.data(), stripBytes);
            if (n < 0)
                throw std::runtime_error("TIFFReadEncodedStrip failed");
            if (needSwap16)
                swap_uint16_buf(stripbuf.data(),
                                static_cast<size_t>(n) / 2);
            prevStrip = stripIdx;
        }

        const uint32_t relRow  = imgY - stripIdx * rowsPerStrip;
        const uint8_t* scanPtr = stripbuf.data() +
            static_cast<size_t>(relRow) * imgW * bytesPerPixel;

        for (int c = 0; c < task.cropW; ++c) {
            const size_t srcOff = static_cast<size_t>(task.in_col0 + c) * bytesPerPixel;
            const size_t dstOff = (static_cast<size_t>(r) * task.cropW + c) * bytesPerPixel;
            std::memcpy(dst + dstOff, scanPtr + srcOff, bytesPerPixel);
        }
    }
}

// ---------------------------------------------------------------------------
//  Worker thread main
// ---------------------------------------------------------------------------
void workerMain(const std::vector<LoadTask>& tasks,
                std::vector<std::vector<uint8_t>>& results,
                uint8_t                     bpp,
                std::atomic<bool>&          abortFlag,
                std::mutex&                 errMtx,
                std::vector<std::string>&   errors,
                size_t                      first,
                size_t                      last)
{
    for (size_t i = first; i < last && !abortFlag.load(std::memory_order_acquire); ++i) {
        const LoadTask& t = tasks[i];
        try {
            TiffHandle tif(TIFFOpen(t.path.c_str(), "rb"));
            if (!tif)
                throw std::runtime_error("Cannot open file: " + t.path);

            readSubRegionToBuffer(t, tif.get(), bpp, results[i].data());
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(errMtx);
            errors.emplace_back(t.path + ": " + ex.what());
            abortFlag.store(true, std::memory_order_release);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
//  mexFunction (entry point)
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transposeFlag])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First argument must be a cell array of filenames");

    const bool transpose =
        (nrhs == 6) && mxIsLogicalScalarTrue(prhs[5]);

    // ------------------------------------------------------------------ filenames
    const int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
    for (int i = 0; i < numSlices; ++i) {
        MatlabString mstr(mxGetCell(prhs[0], i));
        fileList[i] = mstr.get();
        if (fileList[i].empty())
            mexErrMsgIdAndTxt("load_bl_tif:Input",
                "Filename in cell %d is empty", i + 1);
    }

    // ------------------------------------------------------------------ ROI params
    const int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    const int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    const int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW  = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI", "ROI parameters invalid");

    // ------------------------------------------------------------------ probe slice 0
    TIFFSetWarningHandler(nullptr);
    TiffHandle tif0(TIFFOpen(fileList[0].c_str(), "rb"));
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
            "Cannot open file %s (slice 0)", fileList[0].c_str());

    uint32_t imgW = 0, imgH = 0;
    uint16_t bits = 0, spp = 1;
    TIFFGetField(tif0.get(), TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif0.get(), TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif0.get(), TIFFTAG_BITSPERSAMPLE , &bits);
    TIFFGetFieldDefaulted(tif0.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);

    if (spp != 1 || (bits != 8 && bits != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
            "Only 8/16-bit single-channel TIFFs are supported.");

    const uint8_t bytesPerPixel  = (bits == 16) ? 2 : 1;
    const mxClassID outClass     = (bits == 16) ? mxUINT16_CLASS : mxUINT8_CLASS;
    const mwSize outH            = transpose ? roiW : roiH;
    const mwSize outW            = transpose ? roiH : roiW;

    // ------------------------------------------------------------------ output array
    const mwSize dims[3] = {outH, outW, static_cast<mwSize>(numSlices)};
    mxArray* outArr = mxCreateNumericArray(3, dims, outClass, mxREAL);
    plhs[0] = outArr;
    void* outDataRaw = mxGetData(outArr);
    const size_t pixelsPerSlice = static_cast<size_t>(outH) * outW;
    std::memset(outDataRaw, 0, pixelsPerSlice * numSlices * bytesPerPixel);

    // ------------------------------------------------------------------ task build
    std::vector<LoadTask> tasks;
    std::vector<std::vector<uint8_t>> results;
    std::vector<std::string> errors;
    std::mutex errMtx;

    tasks.reserve(numSlices);
    results.reserve(numSlices);

    for (int z = 0; z < numSlices; ++z) {
        TiffHandle tif(TIFFOpen(fileList[z].c_str(), "rb"));
        if (!tif) {
            errors.emplace_back("Cannot open file " + fileList[z]);
            continue;
        }
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgW);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgH);

        const int img_y_s = std::clamp(roiY0, 0, static_cast<int>(imgH) - 1);
        const int img_x_s = std::clamp(roiX0, 0, static_cast<int>(imgW) - 1);
        const int img_y_e = std::clamp(roiY0 + roiH - 1, 0, static_cast<int>(imgH) - 1);
        const int img_x_e = std::clamp(roiX0 + roiW - 1, 0, static_cast<int>(imgW) - 1);

        const int cropH = img_y_e - img_y_s + 1;
        const int cropW = img_x_e - img_x_s + 1;
        if (cropH <= 0 || cropW <= 0) {
            errors.emplace_back("Slice " + std::to_string(z) + " has no overlap with ROI");
            continue;
        }

        const int out_row0 = img_y_s - roiY0;
        const int out_col0 = img_x_s - roiX0;
        tasks.push_back({
            img_y_s, img_x_s,
            out_row0, out_col0,
            cropH, cropW,
            roiH, roiW,
            z,
            pixelsPerSlice,
            fileList[z],
            transpose
        });
        results.emplace_back(static_cast<size_t>(cropH * cropW * bytesPerPixel));
    }

    // ------------------------------------------------------------------ thread pool
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned nThreads = std::min<unsigned>(hw, static_cast<unsigned>(tasks.size()));

    if (const char* env = std::getenv("LOAD_BL_TIF_THREADS"))
        nThreads = std::clamp<unsigned>(std::atoi(env), 1, tasks.size());

    std::atomic<bool> abortFlag(false);
    std::vector<std::thread> pool;
    pool.reserve(nThreads);

    if (!tasks.empty()) {
        const size_t chunk = (tasks.size() + nThreads - 1) / nThreads;
        for (unsigned t = 0; t < nThreads; ++t) {
            const size_t first = t * chunk;
            const size_t last  = std::min(tasks.size(), first + chunk);
            if (first >= last) break;
            pool.emplace_back(workerMain,
                              std::cref(tasks),
                              std::ref(results),
                              bytesPerPixel,
                              std::ref(abortFlag),
                              std::ref(errMtx),
                              std::ref(errors),
                              first, last);
        }
        for (auto& th : pool) th.join();
    }

    if (!errors.empty()) {
        std::string msg("Errors during load_bl_tif:\n");
        for (const auto& e : errors) msg += e + '\n';
        mexErrMsgIdAndTxt("load_bl_tif:LoadError", "%s", msg.c_str());
    }

    // ------------------------------------------------------------------ blit to mxArray
    uint8_t* outData = static_cast<uint8_t*>(outDataRaw);
    // Copy each result into final output with transpose handling
    for (size_t i = 0; i < tasks.size(); ++i) {
        const LoadTask& t = tasks[i];
        const auto& buf   = results[i];
        for (int r = 0; r < t.cropH; ++r) {
            const size_t dstRowBase =
                computeDstIndex(t, r, 0) * bytesPerPixel;
            const size_t srcRowBase =
                static_cast<size_t>(r) * t.cropW * bytesPerPixel;
            std::memcpy(outData + dstRowBase,
                        buf.data() + srcRowBase,
                        static_cast<size_t>(t.cropW) * bytesPerPixel);
        }
    }
}
// ============================================================================
//  end of file
// ============================================================================
