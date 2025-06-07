// ============================================================================
//  load_bl_tif.cpp  – Fast sub-region TIFF loader for MATLAB
//
//  Final cross-platform build (2025-06-07)
//     • tile & strip caching
//     • TIFFSwabArrayOfShort use
//     • 64-bit-safe indexing
//     • correct MATLAB column-major blit
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
#if !defined(__cpp_lib_clamp) &&      \
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

// ----------------------------- I/O helpers ----------------------------------
inline void swap_uint16_buf(void* buf, size_t n) noexcept
{
    TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(buf),
                         static_cast<tmsize_t>(n));
}

static void readSubRegionToBuffer(const LoadTask& task,
                                  TIFF* tif,
                                  uint8_t bpp,
                                  uint8_t* dst)        // cropH*cropW*bpp
{
    uint32_t imgW=0, imgH=0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
    const bool needSwap16 = (bpp == 2) && TIFFIsByteSwapped(tif);

    // ---------------- tiled -----------------
    if (TIFFIsTiled(tif)) {
        uint32_t tileW=0, tileH=0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (!tileW || !tileH) throw std::runtime_error("Bad tile dims");

        const tmsize_t tileSize = TIFFTileSize(tif);
        if (tileSize <= 0) throw std::runtime_error("Invalid TIFFTileSize()");
        std::vector<uint8_t> tilebuf(static_cast<size_t>(tileSize));
        uint32_t prev = UINT32_MAX;

        for (int r=0; r<task.cropH; ++r) {
            uint32_t y = static_cast<uint32_t>(task.in_row0 + r);
            for (int c=0; c<task.cropW; ++c) {
                uint32_t x = static_cast<uint32_t>(task.in_col0 + c);
                uint32_t tIdx = TIFFComputeTile(tif, x, y, 0, 0);
                if (tIdx != prev) {
                    if (TIFFReadEncodedTile(tif, tIdx,
                            tilebuf.data(), tileSize) < 0)
                        throw std::runtime_error("TIFFReadEncodedTile failed");
                    if (needSwap16)
                        swap_uint16_buf(tilebuf.data(),
                                        static_cast<size_t>(tileSize/2));
                    prev = tIdx;
                }
                uint32_t relY = y % tileH, relX = x % tileW;
                size_t src = (relY*tileW + relX)*bpp;
                size_t dstOff = (static_cast<size_t>(r)*task.cropW + c)*bpp;
                std::memcpy(dst+dstOff, tilebuf.data()+src, bpp);
            }
        }
        return;
    }

    // ---------------- strips ----------------
    uint32_t rps=0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
    if (!rps) rps = imgH;
    const size_t stripBytes = static_cast<size_t>(rps)*imgW*bpp;
    if (!stripBytes || stripBytes > (1u<<31))
        throw std::runtime_error("Strip buffer too big");

    std::vector<uint8_t> stripbuf(stripBytes);
    tstrip_t prev = static_cast<tstrip_t>(-1);

    for (int r=0; r<task.cropH; ++r) {
        uint32_t y = static_cast<uint32_t>(task.in_row0 + r);
        tstrip_t sIdx = TIFFComputeStrip(tif, y, 0);
        if (sIdx != prev) {
            tmsize_t n = TIFFReadEncodedStrip(tif, sIdx,
                                              stripbuf.data(), stripBytes);
            if (n < 0) throw std::runtime_error("ReadEncodedStrip failed");
            if (needSwap16) swap_uint16_buf(stripbuf.data(), n/2);
            prev = sIdx;
        }
        uint32_t rel = y - sIdx*rps;
        const uint8_t* scan = stripbuf.data() +
               static_cast<size_t>(rel)*imgW*bpp;
        for (int c=0; c<task.cropW; ++c) {
            size_t src = static_cast<size_t>(task.in_col0 + c)*bpp;
            size_t dstOff = (static_cast<size_t>(r)*task.cropW + c)*bpp;
            std::memcpy(dst+dstOff, scan+src, bpp);
        }
    }
}

// ----------------------------- Thread worker --------------------------------
void workerMain(const std::vector<LoadTask>& tasks,
                std::vector<std::vector<uint8_t>>& results,
                uint8_t bpp,
                std::atomic<bool>& abortFlag,
                std::mutex& errMtx,
                std::vector<std::string>& errors,
                size_t first, size_t last)
{
    for (size_t i=first;
         i<last && !abortFlag.load(std::memory_order_acquire); ++i)
    {
        const LoadTask& t = tasks[i];
        try {
            TiffHandle tif(TIFFOpen(t.path.c_str(), "rb"));
            if (!tif) throw std::runtime_error("Cannot open file: "+t.path);
            readSubRegionToBuffer(t, tif.get(), bpp, results[i].data());
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> l(errMtx);
            errors.emplace_back(t.path+": "+ex.what());
            abortFlag.store(true, std::memory_order_release);
            return;
        }
    }
}

// ----------------------------- mexFunction ----------------------------------
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs<5 || nrhs>6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, height, width[, transpose])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "First arg must be a cell array of strings");

    const bool transpose =
        (nrhs==6) && mxIsLogicalScalarTrue(prhs[5]);

    // ------------- filenames -------------
    const int numSlices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> fileList(numSlices);
    for (int i=0;i<numSlices;++i){
        MatlabString s(mxGetCell(prhs[0],i));
        if(!*s) mexErrMsgIdAndTxt("load_bl_tif:Input",
                                  "Empty filename at cell %d",i+1);
        fileList[i]=s.get();
    }

    // ------------- ROI -------------------
    const int roiY0 = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    const int roiX0 = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    const int roiH  = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW  = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY0<0||roiX0<0||roiH<1||roiW<1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","Bad ROI params");

    // ------------- probe slice 0 ---------
    TIFFSetWarningHandler(nullptr);
    TiffHandle tif0(TIFFOpen(fileList[0].c_str(),"rb"));
    if(!tif0) mexErrMsgIdAndTxt("load_bl_tif:OpenFail",
               "Cannot open file %s",fileList[0].c_str());
    uint32_t imgW=0,imgH=0; uint16_t bits=0,spp=1;
    TIFFGetField(tif0.get(),TIFFTAG_IMAGEWIDTH ,&imgW);
    TIFFGetField(tif0.get(),TIFFTAG_IMAGELENGTH,&imgH);
    TIFFGetField(tif0.get(),TIFFTAG_BITSPERSAMPLE,&bits);
    TIFFGetFieldDefaulted(tif0.get(),TIFFTAG_SAMPLESPERPIXEL,&spp);
    if (spp!=1||(bits!=8&&bits!=16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
            "Only 8/16-bit grayscale TIFFs supported");

    const uint8_t   bpp       = (bits==16)?2:1;
    const mxClassID outClass  = (bits==16)?mxUINT16_CLASS:mxUINT8_CLASS;
    const mwSize    outH      = transpose?roiW:roiH;
    const mwSize    outW      = transpose?roiH:roiW;
    const size_t    pixPerSlc = static_cast<size_t>(outH)*outW;

    mxArray* outArr = mxCreateNumericArray(3,
                        std::array<mwSize,3>{outH,outW,
                        static_cast<mwSize>(numSlices)}.data(),
                        outClass,mxREAL);
    plhs[0]=outArr;
    void* outDataRaw = mxGetData(outArr);
    std::memset(outDataRaw,0,pixPerSlc*numSlices*bpp);

    // ------------- build tasks -----------
    std::vector<LoadTask> tasks;
    std::vector<std::vector<uint8_t>> results;
    std::vector<std::string> errors;
    std::mutex errMtx;

    tasks.reserve(numSlices);
    results.reserve(numSlices);

    for(int z=0;z<numSlices;++z){
        TiffHandle tif(TIFFOpen(fileList[z].c_str(),"rb"));
        if(!tif){errors.emplace_back("Cannot open "+fileList[z]);continue;}
        TIFFGetField(tif.get(),TIFFTAG_IMAGEWIDTH ,&imgW);
        TIFFGetField(tif.get(),TIFFTAG_IMAGELENGTH,&imgH);
        const int ys = std::clamp(roiY0,0,static_cast<int>(imgH)-1);
        const int xs = std::clamp(roiX0,0,static_cast<int>(imgW)-1);
        const int ye = std::clamp(roiY0+roiH-1,0,static_cast<int>(imgH)-1);
        const int xe = std::clamp(roiX0+roiW-1,0,static_cast<int>(imgW)-1);
        const int cH = ye-ys+1, cW = xe-xs+1;
        if(cH<=0||cW<=0){
            errors.emplace_back("Slice "+std::to_string(z)+" no overlap");
            continue;
        }
        tasks.push_back({
            ys,xs,
            ys-roiY0,xs-roiX0,
            cH,cW,roiH,roiW,
            z,pixPerSlc,fileList[z],transpose});
        results.emplace_back(static_cast<size_t>(cH)*cW*bpp);
    }

    // ------------- thread pool -----------
    const unsigned hw = std::max(1u,std::thread::hardware_concurrency());
    unsigned nThreads = std::min<unsigned>(hw,
                         static_cast<unsigned>(tasks.size()));
    if(const char* e=getenv("LOAD_BL_TIF_THREADS"))
        nThreads = std::clamp<unsigned>(
                     static_cast<unsigned>(std::atoi(e)),
                     1u, static_cast<unsigned>(tasks.size()));

    std::atomic<bool> abort(false);
    std::vector<std::thread> pool;
    pool.reserve(nThreads);
    if(!tasks.empty()){
        size_t chunk=(tasks.size()+nThreads-1)/nThreads;
        for(unsigned t=0;t<nThreads;++t){
            size_t first=t*chunk, last=std::min(tasks.size(),first+chunk);
            if(first>=last)break;
            pool.emplace_back(workerMain,
                std::cref(tasks),std::ref(results),bpp,
                std::ref(abort),std::ref(errMtx),
                std::ref(errors),first,last);
        }
        for(auto& th:pool) th.join();
    }
    if(!errors.empty()){
        std::string msg="Errors during load_bl_tif:\n";
        for(auto& s:errors) msg+=s+'\n';
        mexErrMsgIdAndTxt("load_bl_tif:LoadError","%s",msg.c_str());
    }

    // ------------- blit -------------------
    uint8_t* outData = static_cast<uint8_t*>(outDataRaw);
    for(size_t i=0;i<tasks.size();++i){
        const LoadTask& t=tasks[i];
        const auto& buf = results[i];
        for(int r=0;r<t.cropH;++r){
            for(int c=0;c<t.cropW;++c){
                size_t dst = computeDstIndex(t,r,c)*bpp;
                size_t src = (static_cast<size_t>(r)*t.cropW + c)*bpp;
                std::memcpy(outData+dst,buf.data()+src,bpp);
            }
        }
    }
}
// ============================================================================
//  end of file
// ============================================================================
