/**
 * load_bl_tif.cpp  – single-threaded, clear variable names, fixed ROI column
 *   mex -largeArrayDims CXXFLAGS="$CXXFLAGS -std=c++17" load_bl_tif.cpp -ltiff
 */
#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

using uint8  = unsigned char;
using uint16 = unsigned short;

/* --------------------------------------------------------------------- */
struct LoadTask
{
    std::string filename;
    int  roiY, roiX, roiH, roiW;      // 0-based ROI in TIFF coordinates
    std::size_t zIndex;               // output depth slice
    void*       dstBase;              // MATLAB data pointer
    std::size_t pixelsPerPlane;       // roiW * roiH
    mxClassID   matlabType;
};
/* --------------------------------------------------------------------- */
static void copySubRegion( const LoadTask& task )
{
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if (!tif)
        mexErrMsgIdAndTxt("load_bl_tif:OpenFail","Cannot open %s",task.filename.c_str());

    uint32_t imgW=0,imgH=0; uint16 bits=0,spp=1;
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&imgW);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imgH);
    TIFFGetField(tif,TIFFTAG_BITSPERSAMPLE,&bits);
    TIFFGetFieldDefaulted(tif,TIFFTAG_SAMPLESPERPIXEL,&spp);

    if (spp!=1 || (bits!=8&&bits!=16))
        mexErrMsgIdAndTxt("load_bl_tif:Type","Only 8/16-bit gray TIFFs");

    const std::size_t Bpp = bits/8;
    const std::size_t scanBytes = static_cast<std::size_t>(imgW)*Bpp;
    std::vector<uint8> scan(scanBytes);

    for (int row=0; row<task.roiH; ++row)
    {
        const uint32_t tifRow = static_cast<uint32_t>(task.roiY + row); // 0-based
        if (!TIFFReadScanline(tif,scan.data(),tifRow))
            mexErrMsgIdAndTxt("load_bl_tif:Read","Read row %u",tifRow);

        for (int col=0; col<task.roiW; ++col)
        {
            /* -------- source byte offset in this scan-line -------- */
            const std::size_t srcPixel  = static_cast<std::size_t>(task.roiX + col); // << fixed
            const std::size_t srcOffset = srcPixel * Bpp;

            /* -------- destination byte offset in MATLAB array ----- */
            const std::size_t dstPixel  =
                    static_cast<std::size_t>(col)
                  + static_cast<std::size_t>(row) * task.roiW
                  + task.zIndex * task.pixelsPerPlane;
            const std::size_t dstOffset = dstPixel * Bpp;

            std::memcpy(static_cast<uint8*>(task.dstBase)+dstOffset,
                        scan.data()+srcOffset, Bpp);
        }
    }
    TIFFClose(tif);
}
/* --------------------------------------------------------------------- */
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if (nrhs<5) mexErrMsgIdAndTxt("load_bl_tif:Usage",
        "img = load_bl_tif(files, y, x, height, width)");

    if (!mxIsCell(prhs[0])) mexErrMsgIdAndTxt("load_bl_tif:Input",
        "First arg must be a cell array of filenames");

    const std::size_t depth = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> files(depth);
    for (std::size_t i=0;i<depth;++i){
        char* s = mxArrayToString(mxGetCell(prhs[0],i));
        files[i]=s; mxFree(s);
    }

    /* MATLAB gives 1-based coordinates, convert once to 0-based */
    const int roiY = static_cast<int>(mxGetScalar(prhs[1])) - 1;
    const int roiX = static_cast<int>(mxGetScalar(prhs[2])) - 1;
    const int roiH = static_cast<int>(mxGetScalar(prhs[3]));
    const int roiW = static_cast<int>(mxGetScalar(prhs[4]));
    if (roiY<0||roiX<0||roiH<1||roiW<1)
        mexErrMsgIdAndTxt("load_bl_tif:ROI","ROI parameters invalid");

    /* Inspect first file for bit-depth */
    TIFF* t0 = TIFFOpen(files[0].c_str(),"r");
    if(!t0) mexErrMsgIdAndTxt("load_bl_tif:OpenFail","Cannot open %s",files[0].c_str());
    uint16 bits=0,spp=1; TIFFGetField(t0,TIFFTAG_BITSPERSAMPLE,&bits);
    TIFFGetFieldDefaulted(t0,TIFFTAG_SAMPLESPERPIXEL,&spp); TIFFClose(t0);
    if (spp!=1||(bits!=8&&bits!=16))
        mexErrMsgIdAndTxt("load_bl_tif:Type","Only 8/16-bit gray TIFFs");

    const mxClassID mType=(bits==8)?mxUINT8_CLASS:mxUINT16_CLASS;
    const std::size_t Bpp = bits/8;

    /* Output array: [width height depth] */
    mwSize dims[3]={static_cast<mwSize>(roiW),
                    static_cast<mwSize>(roiH),
                    static_cast<mwSize>(depth)};
    plhs[0]=mxCreateNumericArray(3,dims,mType,mxREAL);
    void* out=mxGetData(plhs[0]);
    const std::size_t planePix = static_cast<std::size_t>(roiW)*roiH;

    /* Copy each slice */
    for (std::size_t z=0; z<depth; ++z)
        copySubRegion({files[z],roiY,roiX,roiH,roiW,
                       z,out,planePix,mType});
}
