#include "mex.h"
#include "tiffio.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

typedef unsigned char  uint8_T;
typedef unsigned short uint16_T;

struct LoadTask {
    std::string filename;
    int  y, x, height, width;   // NOTE: x,y are 1-based coming from MATLAB
    size_t zindex;
    void* dst;
    size_t planeStride;         // width*height  ( because we store [W,H,Z] )
    mxClassID type;
};

static void load_subregion(const LoadTask& task)
{
    TIFF* tif = TIFFOpen(task.filename.c_str(), "r");
    if(!tif) mexErrMsgIdAndTxt("TIFFLoad:OpenFail","Cannot open %s",task.filename.c_str());

    uint32_t imgW,imgH; uint16_t bps,spp=1;
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH ,&imgW);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&imgH);
    TIFFGetField(tif,TIFFTAG_BITSPERSAMPLE,&bps);
    TIFFGetFieldDefaulted(tif,TIFFTAG_SAMPLESPERPIXEL,&spp);

    if(spp!=1)           mexErrMsgIdAndTxt("TIFFLoad:Grayscale","Only grayscale supported");
    if(bps!=8&&bps!=16)  mexErrMsgIdAndTxt("TIFFLoad:BPS","Only 8/16-bit supported");
    if(task.x+task.width -1 > (int)imgW ||
       task.y+task.height-1 > (int)imgH)  mexErrMsgIdAndTxt("TIFFLoad:Bounds","sub-region OOB");

    const size_t pixelSize   = bps/8;      // 1 or 2
    const size_t scanlineLen = imgW*pixelSize;
    std::vector<uint8_t> row(scanlineLen);

    for(int r=0;r<task.height;++r)
    {
        const int tifRow = task.y - 1 + r;                 // 0-based row
        if(!TIFFReadScanline(tif,row.data(),tifRow))
            mexErrMsgIdAndTxt("TIFFLoad:Read","scanline read failed");

        for(int c=0;c<task.width;++c)
        {
            size_t srcPixel    = static_cast<size_t>(task.x - 1 + c);   // 0-based col  ★ FIXED ★
            size_t srcIdx      = srcPixel * pixelSize;

            size_t dstPixelOff = static_cast<size_t>(r)                 // y within block
                               + static_cast<size_t>(c)*task.height     // x  (transposed)
                               + task.zindex*task.planeStride;
            size_t dstByteOff  = dstPixelOff*pixelSize;

            std::memcpy(static_cast<uint8_t*>(task.dst)+dstByteOff,
                        row.data()+srcIdx,
                        pixelSize);
        }
    }
    TIFFClose(tif);
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    if(nrhs<5) mexErrMsgIdAndTxt("TIFFLoad:Usage",
        "Usage: img = load_bl_tif(files, y, x, height, width)");

    if(!mxIsCell(prhs[0])) mexErrMsgIdAndTxt("TIFFLoad:Input",
        "First arg must be cell array of filenames");

    /* gather file list --------------------------------------------------- */
    const size_t nslices = mxGetNumberOfElements(prhs[0]);
    std::vector<std::string> files(nslices);
    for(size_t i=0;i<nslices;++i){
        char* s = mxArrayToString(mxGetCell(prhs[0],i));
        if(!s||!*s) mexErrMsgIdAndTxt("TIFFLoad:BadName","Empty filename");
        files[i]=s; mxFree(s);
    }

    int y      = (int)mxGetScalar(prhs[1]);
    int x      = (int)mxGetScalar(prhs[2]);
    int height = (int)mxGetScalar(prhs[3]);
    int width  = (int)mxGetScalar(prhs[4]);

    /* probe first slice to decide output type ---------------------------- */
    TIFF* tif = TIFFOpen(files[0].c_str(),"r");
    if(!tif) mexErrMsgIdAndTxt("TIFFLoad:OpenFail","Cannot open %s",files[0].c_str());
    uint16_t bps; TIFFGetField(tif,TIFFTAG_BITSPERSAMPLE,&bps); TIFFClose(tif);
    if(bps!=8 && bps!=16) mexErrMsgIdAndTxt("TIFFLoad:BPS","Only 8/16-bit supported");
    mxClassID outType = (bps==8)?mxUINT8_CLASS:mxUINT16_CLASS;

    /* create MATLAB output array in [W H Z] layout ----------------------- */
    mwSize dims[3]={(mwSize)width,(mwSize)height,(mwSize)nslices};
    plhs[0]=mxCreateNumericArray(3,dims,outType,mxREAL);
    void*   outData    = mxGetData(plhs[0]);
    size_t  planeStride= (size_t)width*height;

    /* sequential (single-thread) load ------------------------------------ */
    for(size_t z=0;z<nslices;++z){
        LoadTask t{files[z],y,x,height,width,z,outData,planeStride,outType};
        load_subregion(t);
    }
}
