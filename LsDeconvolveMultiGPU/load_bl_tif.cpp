// ============================================================================
//  load_bl_tif.cpp  – Fast sub-region TIFF loader for MATLAB
//
//  Patched cross-platform build (2025-06-07)
//     • tile & strip caching
//     • row-wise memcpy blit (vs. per-pixel)
//     • TIFFSwabArrayOfShort use (+ optional AVX2 swap)
//     • 64-bit-safe indexing
//     • thread-local TIFF handle cache
//     • static thread-pool guard
//     • C++14 fallback for std::clamp  ✅
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
#include <cstdlib>   // getenv / atoi

// ----------------------------- C++14 clamp back-port ------------------------
#if !defined(__cpp_lib_clamp) &&                                                   \
    (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L) && (__cplusplus < 201703L)
namespace std {
template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}
} // namespace std
#endif
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
    int  in_row0, in_col0;  // ROI origin inside TIFF
    int  out_row0, out_col0;// start in output
    int  cropH,  cropW;     // intersecting size
    int  roiH,   roiW;      // full requested ROI
    int  zIndex;
    size_t pixelsPerSlice;
    std::string path;
    bool transpose;
};

struct TiffCloser {
    void operator()(TIFF* t) const noexcept { if (t) TIFFClose(t); }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

// MATLAB is column-major
inline size_t computeDstIndex(const LoadTask& t, int r, int c) noexcept
{
    if (!t.transpose) {
        return static_cast<size_t>(t.out_row0 + r)
             + static_cast<size_t>(t.out_col0 + c) * t.roiH
             + static_cast<size_t>(t.zIndex)       * t.pixelsPerSlice;
    }
    return static_cast<size_t>(t.out_col0 + c)
         + static_cast<size_t>(t.out_row0 + r) * t.roiW
         + static_cast<size_t>(t.zIndex)       * t.pixelsPerSlice;
}

// ----------------------------- optional AVX2 swap ---------------------------
#if defined(__AVX2__) || (defined(_M_AMD64) && defined(__AVX2__))
#include <immintrin.h>
static inline void swap_uint16_avx(void* buf, size_t n) noexcept
{
    auto* p = reinterpret_cast<__m256i*>(buf);
    size_t vec   = n / 16;          // 16 uint16_t per 256-bit vector
    size_t tail  = n % 16;
    for (size_t i = 0; i < vec; ++i) {
        __m256i v = _mm256_loadu_si256(p + i);
        v = _mm256_or_si256(_mm256_slli_epi16(v, 8), _mm256_srli_epi16(v, 8));
        _mm256_storeu_si256(p + i, v);
    }
    if (tail) {
        uint16_t* t = reinterpret_cast<uint16_t*>(p + vec);
        for (size_t i = 0; i < tail; ++i)
            t[i] = static_cast<uint16_t>((t[i] >> 8) | (t[i] << 8));
    }
}
#endif

inline void swap_uint16_buf(void* buf, size_t n) noexcept
{
#if defined(__AVX2__) || (defined(_M_AMD64) && defined(__AVX2__))
    if (n >= 32) return swap_uint16_avx(buf, n);
#endif
    TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(buf),
                         static_cast<tmsize_t>(n));
}

// ---------------------------------------------------------------------------
//  Thread-local TIFF cache
// ---------------------------------------------------------------------------
struct ThreadCtx {
    std::string lastPath;
    TiffHandle  tif;
};
thread_local ThreadCtx g_tctx;

static TIFF* openCached(const std::string& path)
{
    if (g_tctx.lastPath != path) {
        g_tctx.tif.reset(TIFFOpen(path.c_str(), "rb8")); // BigTIFF-capable
        if (!g_tctx.tif)
            throw std::runtime_error("Cannot open file: " + path);
        g_tctx.lastPath = path;
    }
    return g_tctx.tif.get();
}

// ---------------------------------------------------------------------------
//  I/O helpers
// ---------------------------------------------------------------------------
static void readSubRegionToBuffer(const LoadTask& task,
                                  TIFF* tif,
                                  uint8_t bpp,
                                  uint8_t* dst)        // cropH*cropW*bpp
{
    uint32_t imgW = 0, imgH = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
    const bool needSwap16 = (bpp == 2) && TIFFIsByteSwapped(tif);

    // ---------------- tiled -----------------
    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0, tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (!tileW || !tileH)
            throw std::runtime_error("Bad tile dimensions");

        const tmsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0)
            throw std::runtime_error("Invalid TIFFTileSize()");
        std::vector<uint8_t> tilebuf(static_cast<size_t>(tileSize));
        uint32_t prev = UINT32_MAX;

        for (int r = 0; r < task.cropH; ++r) {
            uint32_t y = static_cast<uint32_t>(task.in_row0 + r);
            uint32_t relY, relX;
            for (int cchunk = 0; cchunk < task.cropW; ) {
                uint32_t x = static_cast<uint32_t>(task.in_col0 + cchunk);
                uint32_t tIdx = TIFFComputeTile(tif, x, y, 0, 0);
                if (tIdx != prev) {
                    if (TIFFReadEncodedTile(tif, tIdx,
                                            tilebuf.data(), tileSize) < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed");
                    if (needSwap16)
                        swap_uint16_buf(tilebuf.data(),
                                        static_cast<size_t>(tileSize / 2));
                    prev = tIdx;
                }
                relY = y % tileH;
                relX = x % tileW;

                // how many pixels can we copy contiguously from this tile?
                uint32_t maxRun = std::min<uint32_t>(tileW - relX,
                                                     task.cropW - cchunk);
                const uint8_t* src = tilebuf.data()
                                   + ((relY * tileW + relX) * bpp);
                uint8_t*       dstRow = dst
                                   + (static_cast<size_t>(r) * task.cropW
                                      + cchunk) * bpp;
                std::memcpy(dstRow, src, static_cast<size_t>(maxRun) * bpp);
                cchunk += maxRun;
            }
        }
        return;
    }

    // ---------------- strips ----------------
    uint32_t rps = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
    if (!rps) rps = imgH;
    const size_t stripBytes = static_cast<size_t>(rps) * imgW * bpp;
    if (!stripBytes || stripBytes > (1uLL << 31))
        throw std::runtime_error("Strip buffer too big");

    std::vector<uint8_t> stripbuf(stripBytes);
    tstrip_t prev = static_cast<tstrip_t>(-1);

    const size_t rowBytes = static_cast<size_t>(task.cropW) * bpp;

    for (int r = 0; r < task.cropH; ++r) {
        uint32_t y = static_cast<uint32_t>(task.in_row0 + r);
        tstrip_t sIdx = TIFFComputeStrip(tif, y, 0);
        if (sIdx != prev) {
            tmsize_t n = TIFFReadEncodedStrip(tif, sIdx,
                                              stripbuf.data(), stripBytes);
            if (n < 0)
                throw std::runtime_error("TIFFReadEncodedStrip failed");
            if (needSwap16) swap_uint16_buf(stripbuf.data(), n / 2);
            prev = sIdx;
        }
        uint32_t rel = y - sIdx * rps;
        const uint8_t* scan = stripbuf.data()
                            + static_cast<size_t>(rel) * imgW * bpp;
        const uint8_t* src = scan
                           + static_cast<size_t>(task.in_col0) * bpp;
        uint8_t* dstRow = dst
                        + static_cast<size_t>(r) * rowBytes;
        std::memcpy(dstRow, src, rowBytes);
    }
}

// ---------------------------------------------------------------------------
//  Simple static thread pool (lifetime = MATLAB session)
// ---------------------------------------------------------------------------
class MexThreadPool {
public:
    static MexThreadPool& instance()
    {
        static MexThreadPool inst;
        return inst;
    }

    template <typename F>
    void parallel_for(size_t nTasks, F&& func)
    {
        if (nThreads_ == 1 || nTasks == 1) {
            for (size_t i = 0; i < nTasks; ++i) func(i);
            return;
        }
        std::atomic_size_t next(0);
        std::vector<std::exception_ptr> errs(nThreads_);

        auto body = [&](unsigned tid)
        {
            try {
                for (;;) {
                    size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= nTasks) break;
                    func(i);
                }
            }
            catch (...) { errs[tid] = std::current_exception(); }
        };

        for (unsigned t = 0; t < nThreads_; ++t)
            threads_[t] = std::thread(body, t);
        for (auto& th : threads_) th.join();

        for (auto& e : errs)
            if (e) std::rethrow_exception(e);
    }

    unsigned size() const noexcept { return nThreads_; }

private:
    MexThreadPool()
    {
        nThreads_ = std::max(1u, std::thread::hardware_concurrency());
        if (const char* e = std::getenv("LOAD_BL_TIF_THREADS"))
            nThreads_ = std::clamp<unsigned>(std::atoi(e), 1u, nThreads_);
        threads_.resize(nThreads_);
        mexLock();                       // keep MEX locked for life of pool
        mexAtExit(+[](void*) { MexThreadPool::instance().~MexThreadPool(); });
    }
    ~MexThreadPool()
    {
        mexUnlock();
    }

    unsigned nThreads_;
    std::vector<std::thread> threads_;
};

// ---------------------------------------------------------------------------
//  mexFunction – entry point
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transpose])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First arg must be a cell array of strings");

    const bool transpose =
        (nrhs == 6) && mxIsLogicalScalarTrue(prhs[5]);

    // ------------- filenames -------------
    const int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
    for (int i = 0; i < numSlices; ++i) {
        MatlabString s(mxGetCell(prhs[0], i));
        if (!*s)
            mexErrMsgIdAndTxt("load_bl_tif:Input",
                              "Empty filename at cell %d", i + 1);
        fileList[i] = s.get();
    }

    // ------------- ROI -------------------
    const int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    const int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    const int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW  = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY0 < 0 || roiX0 < 0 || roiH < 1 || roiW < 1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI", "Bad ROI params");

    // ------------- probe slice 0 ---------
    TIFFSetWarningHandler(nullptr);      // silence libtiff chatter
    TiffHandle tif0(TIFFOpen(fileList[0].c_str(), "rb8"));
    if (!tif0)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
            "Cannot open file %s", fileList[0].c_str());
    uint32_t imgW = 0, imgH = 0; uint16_t bits = 0, spp = 1;
    TIFFGetField(tif0.get(), TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif0.get(), TIFFTAG_IMAGELENGTH, &imgH);
    TIFFGetField(tif0.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif0.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    if (spp != 1 || (bits != 8 && bits != 16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
            "Only 8/16-bit grayscale TIFFs supported");

    const uint8_t   bpp       = (bits == 16) ? 2 : 1;
    const mxClassID outClass  = (bits == 16) ? mxUINT16_CLASS : mxUINT8_CLASS;
    const mwSize    outH      = transpose ? roiW : roiH;
    const mwSize    outW      = transpose ? roiH : roiW;
    const size_t    pixPerSlc = static_cast<size_t>(outH) * outW;

    // ------------------------------------------------------------------ output array
    mwSize dims[3] = { outH, outW, static_cast<mwSize>(numSlices) };
    mxArray* outArr = mxCreateNumericArray(3, dims, outClass, mxREAL);
    plhs[0] = outArr;
    void* outDataRaw = mxGetData(outArr);
    std::memset(outDataRaw, 0, pixPerSlc * numSlices * bpp);

    // ------------- build tasks -----------
    std::vector<LoadTask> tasks;
    std::vector<std::vector<uint8_t>> results;
    std::vector<std::string> errors;
    std::mutex errMtx;

    tasks.reserve(numSlices);
    results.reserve(numSlices);

    for (int z = 0; z < numSlices; ++z) {
        try {
            TiffHandle tif(TIFFOpen(fileList[z].c_str(), "rb8"));
            if (!tif)
                throw std::runtime_error("Cannot open");

            TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH , &imgW);
            TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &imgH);
            const int ys = std::clamp(roiY0, 0, static_cast<int>(imgH) - 1);
            const int xs = std::clamp(roiX0, 0, static_cast<int>(imgW) - 1);
            const int ye = std::clamp(roiY0 + roiH - 1,
                                      0, static_cast<int>(imgH) - 1);
            const int xe = std::clamp(roiX0 + roiW - 1,
                                      0, static_cast<int>(imgW) - 1);
            const int cH = ye - ys + 1, cW = xe - xs + 1;
            if (cH <= 0 || cW <= 0)
                throw std::runtime_error("ROI has no overlap");

            tasks.push_back({
                ys, xs,
                ys - roiY0, xs - roiX0,
                cH, cW, roiH, roiW,
                z, pixPerSlc, fileList[z], transpose });
            results.emplace_back(static_cast<size_t>(cH) * cW * bpp);
        }
        catch (const std::exception& ex) {
            errors.emplace_back(fileList[z] + ": " + ex.what());
        }
    }

    if (!errors.empty())
        mexErrMsgIdAndTxt("load_bl_tif:Init",
            "Errors preparing tasks:\n%s", errors[0].c_str());

    // ------------- run worker tasks ------
    auto& pool = MexThreadPool::instance();
    const size_t nTasks = tasks.size();
    std::atomic<bool> abort(false);

    try {
        pool.parallel_for(nTasks, [&](size_t i)
        {
            const LoadTask& t = tasks[i];
            TIFF* tif = openCached(t.path);
            readSubRegionToBuffer(t, tif, bpp, results[i].data());
        });
    }
    catch (const std::exception& ex) {
        mexErrMsgIdAndTxt("load_bl_tif:LoadError", "%s", ex.what());
    }

    // ------------- blit ------------------
    uint8_t* outData = static_cast<uint8_t*>(outDataRaw);
    for (size_t i = 0; i < tasks.size(); ++i) {
        const LoadTask& t = tasks[i];
        const auto& buf   = results[i];
        const size_t rowBytes = static_cast<size_t>(t.cropW) * bpp;
        for (int r = 0; r < t.cropH; ++r) {
            const uint8_t* src = buf.data() + rowBytes * r;
            uint8_t* dst = outData + computeDstIndex(t, r, 0) * bpp;
            std::memcpy(dst, src, rowBytes);
        }
    }
}
// ============================================================================
//  end of file
// ============================================================================
