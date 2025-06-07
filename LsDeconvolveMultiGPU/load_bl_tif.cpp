// ============================================================================
//  load_bl_tif.cpp   –   safe, threaded sub-region loader for 8/16-bit TIFFs
// ============================================================================
//
//  2025-06-07  •  “hard-cap” patch:   fixes negative-ROI crash, 53 GiB OOM,
//                                     and eliminates per-slice staging buffers
//
//  Compile:  mex -v CXXFLAGS="\$CXXFLAGS -std=c++14" load_bl_tif.cpp -ltiff
// ============================================================================

#include "mex.h"
#include "tiffio.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------- constants -------------------------------------------------------
constexpr uint16_t  kBit8  = 8;
constexpr uint16_t  kBit16 = 16;
constexpr uint64_t  MAX_TIFF_BLOCK_BYTES = 1ull << 30; // 1 GiB sanity
constexpr uint64_t  HARD_CAP             = 8ull << 30; // 8 GiB worst-case cap

// ---------- helpers ---------------------------------------------------------
struct MatlabUTF8 {
    char* p;
    explicit MatlabUTF8(const mxArray* a) : p(mxArrayToUTF8String(a)) {
        if (!p) mexErrMsgIdAndTxt("load_bl_tif:BadString",
                                  "Failed to convert mxArray to UTF-8.");
    }
    ~MatlabUTF8() { mxFree(p); }
    operator const char*() const { return p; }
};

struct TiffCloser { void operator()(TIFF* t) const { if (t) TIFFClose(t); } };
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

inline size_t dstIndex(                // MATLAB column-major
        size_t row, size_t col, size_t slice,
        size_t roiH, size_t roiW, size_t pps,
        bool   transpose) noexcept
{
    return (!transpose)
           ? row + col * roiH + slice * pps
           : col + row * roiW + slice * pps;
}

// ---------- overflow-safe multiply -----------------------------------------
inline uint64_t safeMul(uint64_t a, uint64_t b, const char* msg)
{
    if (a && b > std::numeric_limits<uint64_t>::max() / a)
        mexErrMsgIdAndTxt("load_bl_tif:Overflow", "%s", msg);
    return a * b;
}

// ---------- core block reader (tile + strip) -------------------------------
static void readSubRegion(
        const TIFF* tif,
        uint32_t    y0,  uint32_t x0,
        uint32_t    h,   uint32_t w,
        uint8_t     bpp,
        std::vector<uint8_t>& buf)
{
    uint32_t imgW, imgH;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH , &imgW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);

    const bool tiled = TIFFIsTiled(tif);
    if (tiled)
    {
        uint32_t tW=0, tH=0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH , &tW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tH);
        std::vector<uint8_t> tbuf(
            safeMul(safeMul(tW, tH, "Tile"), bpp,
                    "Tile") );

        uint32_t prevTile = UINT32_MAX;
        for (uint32_t r = 0; r < h; ++r)
        {
            uint32_t imgY = y0 + r;
            for (uint32_t c = 0; c < w; ++c)
            {
                uint32_t imgX   = x0 + c;
                uint32_t tileId = TIFFComputeTile(tif, imgX, imgY, 0, 0);

                if (tileId != prevTile)
                {
                    if (TIFFReadEncodedTile(const_cast<TIFF*>(tif), tileId,
                                             tbuf.data(),
                                             tbuf.size()) < 0)
                        mexErrMsgIdAndTxt("load_bl_tif:Tiled:ReadFail",
                                          "TIFFReadEncodedTile failed.");

                    if (bpp == 2 && TIFFIsByteSwapped(tif))
                        TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(tbuf.data()),
                                             tbuf.size() / 2);

                    prevTile = tileId;
                }

                uint32_t relY   = imgY % tH;
                uint32_t relX   = imgX % tW;
                size_t   srcOff = (relY * tW + relX) * bpp;
                size_t   dstOff = (r * w + c)        * bpp;
                std::memcpy(buf.data() + dstOff, tbuf.data() + srcOff, bpp);
            }
        }
    }
    else
    {
        uint32_t rps = 0; TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
        if (!rps) rps = imgH;
        size_t stripMax = safeMul(safeMul(rps, imgW, "Strip"), bpp, "Strip");
        std::vector<uint8_t> sbuf(stripMax);

        tstrip_t cur = (tstrip_t)-1; tsize_t nbytes = 0;

        for (uint32_t r = 0; r < h; ++r)
        {
            uint32_t imgY   = y0 + r;
            tstrip_t sidx   = TIFFComputeStrip(tif, imgY, 0);
            if (sidx != cur)
            {
                nbytes = TIFFReadEncodedStrip(const_cast<TIFF*>(tif), sidx,
                                              sbuf.data(), stripMax);
                if (nbytes < 0)
                    mexErrMsgIdAndTxt("load_bl_tif:Strip:ReadFail",
                                      "TIFFReadEncodedStrip failed.");
                if (bpp == 2 && TIFFIsByteSwapped(tif))
                    TIFFSwabArrayOfShort(reinterpret_cast<uint16_t*>(sbuf.data()),
                                         nbytes / 2);
                cur = sidx;
            }
            uint32_t rowsDecoded = nbytes / (imgW * bpp);
            uint32_t relRow = imgY - sidx * rps;
            if (relRow >= rowsDecoded)
                mexErrMsgIdAndTxt("load_bl_tif:Strip:Bounds",
                                  "Decoded strip too small.");

            const uint8_t* scan = sbuf.data() + relRow * imgW * bpp;
            std::memcpy(buf.data() + r * w * bpp,
                        scan + x0 * bpp,
                        static_cast<size_t>(w) * bpp);
        }
    }
}

// ---------- worker thread ---------------------------------------------------
static void worker(
        const std::vector<std::string>& files,
        const std::vector<uint32_t>&    y0,
        uint32_t                        x0,
        uint32_t                        h,
        uint32_t                        w,
        uint8_t                         bpp,
        size_t                          roiH,
        size_t                          roiW,
        bool                            transpose,
        void*                           outBase,
        size_t                          pps,
        size_t                          begin,
        size_t                          end,
        std::mutex&                     em,
        std::vector<std::string>&       errs,
        std::atomic<size_t>&            ec)
{
    std::vector<uint8_t> scratch;
    scratch.resize(static_cast<size_t>(h) * w * bpp);

    for (size_t i = begin; i < end; ++i)
    {
        try {
            TiffHandle tif(TIFFOpen(files[i].c_str(), "r"));
            if (!tif)
                throw std::runtime_error("Cannot open " + files[i]);

            readSubRegion(tif.get(), y0[i], x0, h, w, bpp, scratch);

            uint8_t* out = static_cast<uint8_t*>(outBase);
            for (uint32_t r = 0; r < h; ++r)
            {
                for (uint32_t c = 0; c < w; ++c)
                {
                    size_t dst = dstIndex(r, c, i, roiH, roiW, pps, transpose)
                                 * bpp;
                    size_t src = (static_cast<size_t>(r) * w + c) * bpp;
                    std::memcpy(out + dst, scratch.data() + src, bpp);
                }
            }
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(em);
            errs.emplace_back(ex.what());
            ++ec;
        }
    }
}

// ===========================================================================
//                                    MEX
// ===========================================================================
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    // ------------------------------------------------------------ args -----
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("load_bl_tif:Usage",
            "Usage: img = load_bl_tif(files, y, x, h, w[, transpose])");

    if (!mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("load_bl_tif:Input",
            "files must be a cell array of strings.");

    bool transpose = false;
    if (nrhs == 6) transpose = mxIsLogicalScalarTrue(prhs[5]);

    int nslices = static_cast<int>(mxGetNumberOfElements(prhs[0]));
    std::vector<std::string> files(nslices);
    for (int i=0;i<nslices;++i){
        MatlabUTF8 s(mxGetCell(prhs[0],i));
        files[i]=s;
    }

    double y_in=mxGetScalar(prhs[1]);
    double x_in=mxGetScalar(prhs[2]);
    double h_in=mxGetScalar(prhs[3]);
    double w_in=mxGetScalar(prhs[4]);

    if (y_in<1||x_in<1||h_in<1||w_in<1)
        mexErrMsgIdAndTxt("load_bl_tif:Negative",
            "y, x, h, w must be positive (1-based).");

    uint32_t roiY0 = static_cast<uint32_t>(y_in-1);
    uint32_t roiX0 = static_cast<uint32_t>(x_in-1);
    uint32_t roiH  = static_cast<uint32_t>(h_in );
    uint32_t roiW  = static_cast<uint32_t>(w_in );

    // ------------------------------------------------- validate first slice
    uint16_t bps=0, spp=1; uint32_t imgW=0,imgH=0;
    {
        TiffHandle tif(TIFFOpen(files[0].c_str(),"r"));
        if (!tif) mexErrMsgIdAndTxt("load_bl_tif:OpenFail","Cannot open %s",files[0].c_str());
        TIFFGetField(tif.get(),TIFFTAG_IMAGEWIDTH ,&imgW);
        TIFFGetField(tif.get(),TIFFTAG_IMAGELENGTH,&imgH);
        TIFFGetField(tif.get(),TIFFTAG_BITSPERSAMPLE,&bps);
        TIFFGetFieldDefaulted(tif.get(),TIFFTAG_SAMPLESPERPIXEL,&spp);
    }
    if (spp!=1||(bps!=kBit8&&bps!=kBit16))
        mexErrMsgIdAndTxt("load_bl_tif:Type",
            "Only 8/16-bit grayscale TIFFs supported.");

    if (roiY0+roiH>imgH||roiX0+roiW>imgW)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI out of bounds.");

    uint8_t bpp = (bps==16)?2:1;

    // ---------------------------------------------------- memory hard-cap
    uint64_t roiPix = safeMul(roiH, roiW, "ROI");
    uint64_t memNeed = safeMul(roiPix, bpp, "Mem")
                     * static_cast<uint64_t>(nslices);
    bool streaming = memNeed > HARD_CAP;

    // ---------------------------------------------------- output mxArray --
    mwSize outH = transpose ? roiW : roiH;
    mwSize outW = transpose ? roiH : roiW;
    mwSize dims[3]={outH,outW,static_cast<mwSize>(nslices)};
    plhs[0]=mxCreateNumericArray(3,dims,(bps==8)?mxUINT8_CLASS:mxUINT16_CLASS,mxREAL);
    if (!plhs[0]) mexErrMsgIdAndTxt("load_bl_tif:Alloc","mxCreateNumericArray failed");

    void* outData = mxGetData(plhs[0]);
    size_t pps    = static_cast<size_t>(outH)*outW;

    // ----------------------------------------------------------- y0 vector
    std::vector<uint32_t> y0(nslices, roiY0); // same for every slice

    // ----------------------------------------------------------- strategy
    if (streaming)   // ---------- single-thread, low-mem -----------
    {
        std::vector<uint8_t> scratch(roiPix * bpp);

        for (int z=0; z<nslices; ++z)
        {
            TiffHandle tif(TIFFOpen(files[z].c_str(),"r"));
            if (!tif) mexErrMsgIdAndTxt("load_bl_tif:OpenFail","Cannot open %s",files[z].c_str());

            readSubRegion(tif.get(), roiY0, roiX0, roiH, roiW, bpp, scratch);

            uint8_t* out = static_cast<uint8_t*>(outData);
            for (uint32_t r=0; r<roiH; ++r)
            for (uint32_t c=0; c<roiW; ++c)
            {
                size_t dst = dstIndex(r,c,z,roiH,roiW,pps,transpose)*bpp;
                size_t src = (static_cast<size_t>(r)*roiW+c)*bpp;
                std::memcpy(out+dst, scratch.data()+src, bpp);
            }
        }
    }
    else             // ---------- thread-pool fast path ------------
    {
        unsigned nth = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
        if (const char* e=getenv("LOAD_BL_TIF_THREADS"))
            nth = std::max(1u, static_cast<unsigned>(atoi(e)));
#endif
        size_t chunk = (nslices + nth - 1) / nth;

        std::vector<std::thread> th;
        std::mutex em; std::vector<std::string> errs;
        std::atomic<size_t> ec{0};

        for (unsigned t=0; t<nth; ++t)
        {
            size_t b=t*chunk, e=std::min<size_t>(nslices,b+chunk);
            if (b>=e) break;
            th.emplace_back(worker,
                std::cref(files), std::cref(y0),
                roiX0, roiH, roiW, bpp,
                roiH, roiW, transpose,
                outData, pps,
                b, e, std::ref(em), std::ref(errs), std::ref(ec));
        }
        for (auto& w:th) w.join();

        if (ec)
        {
            std::ostringstream oss; oss<<"Errors ("<<ec<<"):\n";
            for (auto& s:errs) oss<<"  - "<<s<<"\n";
            mexErrMsgIdAndTxt("load_bl_tif:Threaded","%s",oss.str().c_str());
        }
    }
}
