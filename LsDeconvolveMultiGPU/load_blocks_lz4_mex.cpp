/*==============================================================================
  load_blocks_lz4_mex.cpp
  -----------------------------------------------------------------------------
  Parallel block-by-block LZ4 loader that reconstructs a very large 3‑D single
  precision volume directly inside MATLAB memory.

  Author:       Keivan Moradi  (initial specification)
  Implemented:  ChatGPT‑4o‑2025‑06‑17
  License:      GPL-v3 (https://www.gnu.org/licenses/gpl-3.0.html)

  OVERVIEW
  --------
  Given a cell‑array of *.lz4c* block files (each created by *save_lz4_mex*),
  together with the XYZ start/end coordinates of each block inside the final
  volume, this MEX function

      • pre‑allocates the destination single‑precision array R  ➜  shared
        between MATLAB and the worker threads;
      • launches a pool of C++11 threads (size = min(#blocks, HW threads));
      • each thread
          – opens / validates the block file header,
          – decompresses its chunked payload with LZ4 directly into a private
            temporary buffer (to keep the codec simple and safe),
          – copies the buffer into the correct position inside R using
            memcpy per‑row (no overlap ⇒ no locks),
          – propagates any error back to the main thread.

  The implementation re‑uses the header specification of *load_lz4_mex.c*.
  The only supported dtype is **single** (32‑bit float) to keep code concise
  and avoid template bloat. It can be extended easily.

  USAGE (MATLAB)
  --------------
      [R, elapsed] = load_blocks_lz4_mex( ...
              filenames,          % {N×1} cellstr
              block_p1,           % [N×3] double/uint64 – 1‑based inclusive
              block_p2,           % [N×3] double/uint64 – 1‑based inclusive
              R_size,             % [1×3] double – [X Y Z]
              max_threads )       % (optional) scalar

      • *filenames*  – absolute or relative paths to .lz4c block files
      • *block_p1/p2* – start & end coordinates (MATLAB 1‑based indexing)
      • *R_size*     – full volume dimensions [X Y Z]
      • *max_threads* – clamp the thread pool (default = hardware_concurrency)

      The function returns the reconstructed volume R and the elapsed loading
      time in seconds.

  COMPILATION
  -----------
      mex -R2018a CXXFLAGS="$CXXFLAGS -std=c++17" \
          load_blocks_lz4_mex.cpp lz4.c

  DEPENDENCIES
  ------------
      – lz4.c / lz4.h (https://github.com/lz4/lz4)

==============================================================================*/

/*==============================================================================
  load_blocks_lz4_mex.cpp   (2025-06-18, edge-brick-safe)
==============================================================================*/

#include "mex.h"
#include "lz4.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#define MAX_DIMS     16
#define MAX_CHUNKS   2048
#define HEADER_SIZE  33280U
#define MAGIC_NUMBER 0x4C5A4331U

enum dtype_enum : uint8_t { DT_DOUBLE=1, DT_SINGLE=2, DT_UINT16=3 };

struct file_header_t {
    uint32_t magic;
    uint8_t  dtype;
    uint8_t  ndims;
    uint64_t dims[MAX_DIMS];
    uint64_t total_uncompressed;
    uint64_t chunk_size;
    uint32_t num_chunks;
    uint64_t chunk_uncomp[MAX_CHUNKS];
    uint64_t chunk_comp  [MAX_CHUNKS];
    uint8_t  padding[HEADER_SIZE-(4+1+1+8*MAX_DIMS+8+8+4+8*MAX_CHUNKS*2)];
};

static inline void fread_exact(FILE* f, void* dst, size_t n, const char* err)
{
    if (std::fread(dst,1,n,f)!=n) throw std::runtime_error(err);
}

/* ★ accept ndims==2 OR 3; trailing dims implicitly 1 */
static file_header_t read_header(FILE* f)
{
    file_header_t h{};  fread_exact(f,&h,HEADER_SIZE,"read header");
    if (h.magic != MAGIC_NUMBER) throw std::runtime_error("bad magic");
    if (h.dtype != DT_SINGLE)    throw std::runtime_error("not single");
    if (h.ndims!=2 && h.ndims!=3)
        throw std::runtime_error("Block ndims must be 2 or 3");
    if (h.num_chunks==0||h.num_chunks>MAX_CHUNKS)
        throw std::runtime_error("chunk count bad");
    return h;
}

static inline uint64_t lin(uint64_t x,uint64_t y,uint64_t z,
                           uint64_t Dx,uint64_t Dy)
{ return x + Dx*(y + Dy*z); }

/* ------------------------------------------------------------------ */
struct BlockTask {
    std::string fname;
    uint64_t x0,y0,z0, x1,y1,z1;
    uint64_t Dx,Dy,Dz;
    float*   R;

    void operator()() const
    {
        FILE* f=std::fopen(fname.c_str(),"rb");
        if(!f) throw std::runtime_error("open "+fname);
        try{
            const file_header_t h = read_header(f);

            uint64_t bx=x1-x0+1, by=y1-y0+1, bz=z1-z0+1;

            /* ★ adapt header dims (ndims==2 ⇒ z=1) */
            const uint64_t hdrBx = h.dims[0];
            const uint64_t hdrBy = h.dims[1];
            const uint64_t hdrBz = (h.ndims==3) ? h.dims[2] : 1ULL;
            if(hdrBx!=bx||hdrBy!=by||hdrBz!=bz)
                throw std::runtime_error("dim mismatch "+fname);

            std::vector<float> buf(bx*by*bz);
            char* d=reinterpret_cast<char*>(buf.data());
            uint64_t written=0;

            for(uint32_t c=0;c<h.num_chunks;++c){
                uint64_t cs=h.chunk_comp[c], usz=h.chunk_uncomp[c];
                if(cs>0x7FFFFFFF||usz>0x7FFFFFFF)
                    throw std::runtime_error("chunk>2GB "+fname);
                std::vector<char> comp(cs);
                fread_exact(f,comp.data(),(size_t)cs,
                            ("read error "+fname).c_str());
                int dec=LZ4_decompress_safe(comp.data(),d+written,
                                            (int)cs,(int)usz);
                if(dec<0|| (uint64_t)dec!=usz)
                    throw std::runtime_error("LZ4 error "+fname);
                written+=usz;
            }
            if(written!=h.total_uncompressed)
                throw std::runtime_error("size mismatch "+fname);

            const uint64_t row=bx, offY=Dx, offZ=Dx*Dy;
            const float* src=buf.data();
            for(uint64_t z=0;z<bz;++z)
              for(uint64_t y=0;y<by;++y){
                  uint64_t dst=lin(x0,y0+y,z0+z,Dx,Dy);
                  std::memcpy(R+dst,src,row*sizeof(float));
                  src+=row;
              }
            std::fclose(f);
        }catch(...){std::fclose(f);throw;}
    }
};

/* ------------------------------------------------------------------ */
void mexFunction(int nlhs,mxArray*plhs[],int nrhs,const mxArray*prhs[])
{
    auto tic=std::chrono::high_resolution_clock::now();
    if(nrhs<4||nrhs>5)
        mexErrMsgTxt("[R,t]=load_blocks_lz4_mex(fns,p1,p2,Rsz,[maxT])");

    /* filenames */
    if(!mxIsCell(prhs[0])) mexErrMsgTxt("fns cell");
    mwSize N=mxGetNumberOfElements(prhs[0]);
    if(!N) mexErrMsgTxt("empty fns");

    const mxArray *p1mx=prhs[1], *p2mx=prhs[2];
    if(mxGetM(p1mx)!=N||mxGetN(p1mx)!=3||
       mxGetM(p2mx)!=N||mxGetN(p2mx)!=3)
        mexErrMsgTxt("p1/p2 size");

    auto idx=[&](const mxArray*A,mwSize i)->uint64_t{
        return mxIsUint64(A) ? ((uint64_t*)mxGetData(A))[i]
                             : (uint64_t)mxGetPr(A)[i];
    };

    uint64_t Dx,Dy,Dz;
    if(mxIsUint64(prhs[3])){
        auto v=(uint64_t*)mxGetData(prhs[3]);Dx=v[0];Dy=v[1];Dz=v[2];
    }else{
        double*v=mxGetPr(prhs[3]);Dx=v[0];Dy=v[1];Dz=v[2];
    }

    int maxT = (nrhs==5) ? (int)mxGetScalar(prhs[4])
                         : (int)std::thread::hardware_concurrency();
    if(maxT<1) maxT=1;

    mwSize md[3]={(mwSize)Dx,(mwSize)Dy,(mwSize)Dz};
    mxArray* Rmx=mxCreateNumericArray(3,md,mxSINGLE_CLASS,mxREAL);
    if(!Rmx) mexErrMsgTxt("OOM R");
    float* R=(float*)mxGetData(Rmx);

    std::vector<BlockTask> tasks; tasks.reserve(N);
    for(mwSize i=0;i<N;++i){
        char* c=mxArrayToUTF8String(mxGetCell(prhs[0],i));
        std::string fn(c); mxFree(c);

        uint64_t x0=idx(p1mx,i)-1,y0=idx(p1mx,i+N)-1,z0=idx(p1mx,i+2*N)-1;
        uint64_t x1=idx(p2mx,i)-1,y1=idx(p2mx,i+N)-1,z1=idx(p2mx,i+2*N)-1;

        if(x1<x0||y1<y0||z1<z0) mexErrMsgTxt("p1>p2");
        if(x1>=Dx||y1>=Dy||z1>=Dz) mexErrMsgTxt("p2>Rsz");

        tasks.push_back({fn,x0,y0,z0,x1,y1,z1,Dx,Dy,Dz,R});
    }

    std::vector<std::future<void>> futs;
    size_t next=0; auto launch=[&](size_t k){
        futs.emplace_back(std::async(std::launch::async,tasks[k]));
    };
    for(;next<std::min<size_t>(tasks.size(),(size_t)maxT);++next) launch(next);

    while(!futs.empty()){
        for(auto it=futs.begin();it!=futs.end();){
            if(it->wait_for(std::chrono::milliseconds(10))
                 ==std::future_status::ready){
                it->get(); it=futs.erase(it);
                if(next<tasks.size()) launch(next++);
            }else ++it;
        }
    }

    double t=std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now()-tic).count();
    plhs[0]=Rmx;
    if(nlhs>1) plhs[1]=mxCreateDoubleScalar(t);
}
