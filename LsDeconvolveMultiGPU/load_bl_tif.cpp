// ============================================================================
//  load_bl_tif.cpp  – Fast sub-region TIFF loader for MATLAB
//
//  Patched 2025-06-07
//     • tile & strip caching
//     • TIFFSwabArrayOfShort use
//     • 64-bit-safe indexing
//     • thread-pool fixes & misc. clean-ups
//     • correct column-major blit for non-transposed mode  ← NEW
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
//  Helpers & traits
// ---------------------------------------------------------------------------
struct MatlabString {
    char* ptr;
    explicit MatlabString(const mxArray* arr) : ptr(mxArrayToUTF8String(arr)) {
        if (!ptr)
            mexErrMsgIdAndTxt("load_bl_tif:BadString",
                              "Failed to convert MATLAB string");
    }
    ~MatlabString() { mxFree(ptr); }
    const char* get()  const noexcept { return ptr; }
    operator const char*() const noexcept { return ptr; }
};

struct TiffCloser { void operator()(TIFF* t) const noexcept { if (t) TIFFClose(t); } };
using  TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

struct LoadTask {
    // crop position in source TIFF
    int in_row0,  in_col0;
    // where block lands inside destination array
    int out_row0, out_col0;
    // crop size
    int cropH,    cropW;
    // full ROI (for strides)
    int roiH,     roiW;
    // z-slice info
    int zIndex;
    size_t pixelsPerSlice;
    // misc
    std::string path;
    bool transpose;
};

// MATLAB column-major index helper
inline size_t computeDstIndex(const LoadTask& t, int r, int c) noexcept
{
    if (!t.transpose)   // (row, col, z)
        return static_cast<size_t>(t.out_row0 + r)
             + static_cast<size_t>(t.out_col0 + c) * t.roiH
             + static_cast<size_t>(t.zIndex)       * t.pixelsPerSlice;

    // transposed: swap row/col interpretation
    return static_cast<size_t>(t.out_col0 + c)
         + static_cast<size_t>(t.out_row0 + r) * t.roiW
         + static_cast<size_t>(t.zIndex)       * t.pixelsPerSlice;
}

// ---------------------------------------------------------------------------
//  Low-level TIFF helpers
// ---------------------------------------------------------------------------
inline void swab16(void* buf, size_t count) noexcept
{
    TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(buf),
                         static_cast<tmsize_t>(count));
}

// Copy task.cropH × task.cropW pixels into dst (row-major in dst buffer)
static void readSubRegionToBuffer(const LoadTask& task, TIFF* tif,
                                  uint8_t bpp, uint8_t* dst)
{
    uint32_t imgW = 0, imgH = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);

    const bool needSwap = (bpp == 2) && TIFFIsByteSwapped(tif);

    // -------- tiled ---------------------------------------------------------
    if (TIFFIsTiled(tif))
    {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (!tileW || !tileH)
            throw std::runtime_error("Invalid tile dimensions");

        const tmsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0)
            throw std::runtime_error("Invalid TIFFTileSize");

        std::vector<uint8_t> tileBuf(static_cast<size_t>(tileSize));
        uint32_t prevTile = UINT32_MAX;

        for (int r = 0; r < task.cropH; ++r)
        {
            const uint32_t imgY = static_cast<uint32_t>(task.in_row0 + r);
            for (int c = 0; c < task.cropW; ++c)
            {
                const uint32_t imgX = static_cast<uint32_t>(task.in_col0 + c);
                const uint32_t tileIdx = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (tileIdx != prevTile)
                {
                    if (TIFFReadEncodedTile(tif, tileIdx,
                                            tileBuf.data(), tileSize) < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed");

                    if (needSwap) swab16(tileBuf.data(),
                                         static_cast<size_t>(tileSize / 2));
                    prevTile = tileIdx;
                }

                const uint32_t relY = imgY % tileH, relX = imgX % tileW;
                const size_t   src  = (relY * tileW + relX) * bpp;
                const size_t   dstIdx = (static_cast<size_t>(r) * task.cropW + c) * bpp;
                std::memcpy(dst + dstIdx, tileBuf.data() + src, bpp);
            }
        }
        return;
    }

    // -------- strips --------------------------------------------------------
    uint32_t rowsPerStrip = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
    if (!rowsPerStrip) rowsPerStrip = imgH;

    const size_t stripBytes = static_cast<size_t>(rowsPerStrip) * imgW * bpp;
    if (!stripBytes || stripBytes > (1u << 31))
        throw std::runtime_error("Bad strip buffer size");

    std::vector<uint8_t> stripBuf(stripBytes);
    tstrip_t prevStrip = static_cast<tstrip_t>(-1);

    for (int r = 0; r < task.cropH; ++r)
    {
        const uint32_t imgY = static_cast<uint32_t>(task.in_row0 + r);
        const tstrip_t stripIdx = TIFFComputeStrip(tif, imgY, 0);

        if (stripIdx != prevStrip)
        {
            const tmsize_t n = TIFFReadEncodedStrip(tif, stripIdx,
                                                    stripBuf.data(), stripBytes);
            if (n < 0) throw std::runtime_error("TIFFReadEncodedStrip failed");
            if (needSwap) swab16(stripBuf.data(), static_cast<size_t>(n) / 2);
            prevStrip = stripIdx;
        }

        const uint8_t* scan = stripBuf.data() +
            static_cast<size_t>(imgY - stripIdx * rowsPerStrip) * imgW * bpp;

        for (int c = 0; c < task.cropW; ++c)
        {
            const size_t src = static_cast<size_t>(task.in_col0 + c) * bpp;
            const size_t dstIdx = (static_cast<size_t>(r) * task.cropW + c) * bpp;
            std::memcpy(dst + dstIdx, scan + src, bpp);
        }
    }
}

// ---------------------------------------------------------------------------
//  Worker thread
// ---------------------------------------------------------------------------
static void workerMain(const std::vector<LoadTask>& tasks,
                       std::vector<std::vector<uint8_t>>& results,
                       uint8_t bpp,
                       std::atomic<bool>& abortFlag,
                       std::mutex& errMtx,
                       std::vector<std::string>& errors,
                       size_t first, size_t last)
{
    for (size_t i = first;
         i < last && !abortFlag.load(std::memory_order_acquire); ++i)
    {
        const LoadTask& T = tasks[i];
        try {
            TiffHandle tif(TIFFOpen(T.path.c_str(), "rb"));
            if (!tif) throw std::runtime_error("Cannot open file: " + T.path);

            readSubRegionToBuffer(T, tif.get(), bpp, results[i].data());
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(errMtx);
            errors.emplace_back(ex.what());
            abortFlag.store(true, std::memory_order_release);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
//  MEX entry point
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
                          "Usage: img = load_bl_tif(files, y, x, h, w [, transpose])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input", "First arg must be a cell array");

    const bool transpose =
        (nrhs == 6) && mxIsLogicalScalarTrue(prhs[5]);

    // ---------------- read filenames ---------------------------------
    const int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> files(numSlices);
    for (int i = 0; i < numSlices; ++i) {
        MatlabString s(mxGetCell(prhs[0], i));
        files[i] = s.get();
        if (files[i].empty())
            mexErrMsgIdAndTxt("load_bl_tif:Input",
                              "Filename in cell %d is empty", i + 1);
    }

    // ---------------- ROI --------------------------------------------
    const int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    const int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    const int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW  = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI", "ROI parameters invalid");

    // ---------------- probe first slice ------------------------------
    TIFFSetWarningHandler(nullptr);
    TiffHandle tif0(TIFFOpen(files[0].c_str(), "rb"));
    if (!tif0) mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
                                 "Cannot open file %s", files[0].c_str());

    uint32_t imgW = 0, imgH = 0;
    uint16_t bits = 0, spp = 1;
    TIFFGetField(tif0.get(), TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif0.get(), TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif0.get(), TIFFTAG_BITSPERSAMPLE , &bits);
    TIFFGetFieldDefaulted(tif0.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    if (spp != 1 || (bits != 8 && bits != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
                          "Only 8/16-bit single-channel TIFF supported");

    const uint8_t   bpp      = (bits == 16) ? 2 : 1;
    const mxClassID outClass = (bits == 16) ? mxUINT16_CLASS : mxUINT8_CLASS;
    const mwSize    outH     = transpose ? roiW : roiH;
    const mwSize    outW     = transpose ? roiH : roiW;

    // ---------------- output array -----------------------------------
    const mwSize dims[3] = {outH, outW, static_cast<mwSize>(numSlices)};
    mxArray* outArr = mxCreateNumericArray(3, dims, outClass, mxREAL);
    plhs[0] = outArr;
    void*   outDataRaw = mxGetData(outArr);
    const size_t pixelsPerSlice = static_cast<size_t>(outH) * outW;
    std::memset(outDataRaw, 0, pixelsPerSlice * numSlices * bpp);

    // ---------------- build tasks ------------------------------------
    std::vector<LoadTask> tasks;
    std::vector<std::vector<uint8_t>> buffers;
    std::vector<std::string> errors;
    std::mutex errMtx;

    tasks.reserve(numSlices);
    buffers.reserve(numSlices);

    for (int z = 0; z < numSlices; ++z)
    {
        TiffHandle tif(TIFFOpen(files[z].c_str(), "rb"));
        if (!tif) { errors.emplace_back("Cannot open " + files[z]); continue; }

        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgW);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgH);

        const int ys = std::clamp(roiY0, 0, static_cast<int>(imgH) - 1);
        const int xs = std::clamp(roiX0, 0, static_cast<int>(imgW) - 1);
        const int ye = std::clamp(roiY0 + roiH - 1, 0, static_cast<int>(imgH) - 1);
        const int xe = std::clamp(roiX0 + roiW - 1, 0, static_cast<int>(imgW) - 1);

        const int cropH = ye - ys + 1, cropW = xe - xs + 1;
        if (cropH <= 0 || cropW <= 0) {
            errors.emplace_back("Slice " + std::to_string(z) + " no ROI overlap");
            continue;
        }

        tasks.push_back({
            ys, xs,
            ys - roiY0, xs - roiX0,
            cropH, cropW,
            roiH,  roiW,
            z,
            pixelsPerSlice,
            files[z],
            transpose
        });
        buffers.emplace_back(static_cast<size_t>(cropH * cropW * bpp));
    }

    // ---------------- parallel load ----------------------------------
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned nThreads = std::min<unsigned>(hw, tasks.size());
    if (const char* env = std::getenv("LOAD_BL_TIF_THREADS"))
        nThreads = std::clamp<unsigned>(std::atoi(env), 1, tasks.size());

    std::atomic<bool> abortFlag(false);
    std::vector<std::thread> pool;
    pool.reserve(nThreads);

    if (!tasks.empty())
    {
        const size_t chunk = (tasks.size() + nThreads - 1) / nThreads;
        for (unsigned t = 0; t < nThreads; ++t) {
            const size_t first = t * chunk, last = std::min(tasks.size(), first + chunk);
            if (first >= last) break;
            pool.emplace_back(workerMain, std::cref(tasks), std::ref(buffers),
                              bpp, std::ref(abortFlag), std::ref(errMtx),
                              std::ref(errors), first, last);
        }
        for (auto& th : pool) th.join();
    }

    if (!errors.empty()) {
        std::string msg("Errors during load_bl_tif:\n");
        for (const auto& e : errors) msg += e + '\n';
        mexErrMsgIdAndTxt("load_bl_tif:LoadError", "%s", msg.c_str());
    }

    // ---------------- final blit -------------------------------------
    uint8_t* outData = static_cast<uint8_t*>(outDataRaw);

    for (size_t i = 0; i < tasks.size(); ++i)
    {
        const LoadTask& T = tasks[i];
        const auto& buf   = buffers[i];

        if (!T.transpose)  // column-major write, column by column
        {
            for (int c = 0; c < T.cropW; ++c)
            {
                const size_t dstColBase =
                    (static_cast<size_t>(T.out_col0 + c) * T.roiH +
                     T.out_row0) +
                    static_cast<size_t>(T.zIndex) * T.pixelsPerSlice;

                for (int r = 0; r < T.cropH; ++r)
                {
                    const size_t dstIdx = (dstColBase + r) * bpp;
                    const size_t srcIdx = (static_cast<size_t>(r) * T.cropW + c) * bpp;
                    std::memcpy(outData + dstIdx, buf.data() + srcIdx, bpp);
                }
            }
        }
        else              // transposed: rows are contiguous
        {
            for (int r = 0; r < T.cropH; ++r)
            {
                const size_t dstRowBase =
                    computeDstIndex(T, r, 0) * bpp;
                const size_t srcRowBase =
                    static_cast<size_t>(r) * T.cropW * bpp;

                std::memcpy(outData + dstRowBase,
                            buf.data() + srcRowBase,
                            static_cast<size_t>(T.cropW) * bpp);
            }
        }
    }
}
// ============================================================================
//  end of file
// ============================================================================
