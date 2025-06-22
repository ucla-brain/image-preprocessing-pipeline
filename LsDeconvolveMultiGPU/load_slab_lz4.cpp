/*==============================================================================
  load_slab_lz4.cpp
  -------------------------------------------------------------------------------
  Single-precision, threaded loader for LZ4-compressed bricks.
  Performs MATLAB-style rescale / clip inside worker threads and writes the
  final uint8 / uint16 volume in-place.
==============================================================================*/

#include "lz4.h"
#include "matrix.h"
#include "mex.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/* ── portable fmaf fallback ─────────────────────────────────────────── */
#ifndef fmaf   /* some old/embedded toolchains don’t have std::fmaf */
#  define fmaf(a,b,c) ((a)*(b)+(c))
#endif

/* ── per-thread buffers ─────────────────────────────────────────────── */
struct ThreadLocalCleaner {
    ~ThreadLocalCleaner() { thread_u.clear(); thread_c.clear(); }
    static thread_local std::vector<float> thread_u;   // uncompressed voxels
    static thread_local std::vector<char>  thread_c;   // compressed chunk
};
thread_local std::vector<float> ThreadLocalCleaner::thread_u;
thread_local std::vector<char>  ThreadLocalCleaner::thread_c;

/* ── tiny FIFO thread-pool ------------------------------------------- */
class ThreadPool {
public:
    explicit ThreadPool(std::size_t n)
        : stop_(false), pending_(0)
    {
        for (std::size_t i = 0; i < n; ++i)
            workers_.emplace_back([this]{
                ThreadLocalCleaner cleaner;
                while (true) {
                    std::function<void()> job;
                    {   std::unique_lock<std::mutex> lk(m_);
                        cv_job_.wait(lk,[&]{return stop_ || !q_.empty();});
                        if (stop_ && q_.empty()) return;
                        job = std::move(q_.front()); q_.pop();
                    }
                    job();
                    if (pending_.fetch_sub(1) == 1) cv_done_.notify_one();
                }
            });
    }

    template<class F> void enqueue(F&& f) {
        std::lock_guard<std::mutex> lk(m_);
        q_.emplace(std::forward<F>(f));
        pending_.fetch_add(1);
        // Wake all threads, so many can pick up jobs in parallel
        cv_job_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk,[&]{return pending_.load()==0;});
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_=true; }
        cv_job_.notify_all();
        for (auto& t:workers_) t.join();
    }
private:
    std::vector<std::thread>            workers_;
    std::queue<std::function<void()>>   q_;
    std::mutex                          m_;
    std::condition_variable             cv_job_,cv_done_;
    std::atomic<std::size_t>            pending_;
    bool                                stop_;
};

/* ── LZ4 header helpers (unchanged from earlier) ────────────────────── */
constexpr uint32_t MAGIC = 0x4C5A4331U;
constexpr uint32_t HSIZE = 33280U;
constexpr uint32_t MAXC  = 2048U;
enum DType: uint8_t { DT_DOUBLE=1, DT_SINGLE=2, DT_UINT16=3 };
struct Header {
    uint32_t magic; uint8_t dtype,ndims; uint64_t dims[16];
    uint64_t total,chunkSize; uint32_t nChunks;
    uint64_t uLen[MAXC], cLen[MAXC];
    uint8_t  pad[HSIZE-(4+1+1+16*8+8+8+4+MAXC*16)];
};
static void freadExact(FILE* f,void* d,size_t n,const char* ctx){
    if(std::fread(d,1,n,f)!=n) throw std::runtime_error(ctx);
}
static Header readHeader(FILE* f,const std::string& nm){
    Header h{}; freadExact(f,&h,HSIZE,"header");
    if(h.magic!=MAGIC)            throw std::runtime_error(nm+": bad magic");
    if(h.dtype!=DT_SINGLE)        throw std::runtime_error(nm+": not single");
    if(h.nChunks==0||h.nChunks>MAXC) throw std::runtime_error(nm+": bad chunks");
    return h;
}
static inline uint64_t idx3D(uint64_t x,uint64_t y,uint64_t z,
                             uint64_t dx,uint64_t dy){
    return x + dx*(y + dy*z);
}

/* ── BrickJob : 100 % single-precision maths ───────────────────────── */
struct BrickJob {
    /* geometry & dest */
    std::string file; uint64_t x0,y0,z0,x1,y1,z1, dimX,dimY,dimZ; float* vol;
    /* parameters (double in API, cast once to float) */
    float scal, ampl, low_clip, high_clip, dmin, dmax; bool clip;

    void operator()() const {
        /* 1. decompress -------------------------------------------------- */
        auto& U = ThreadLocalCleaner::thread_u;
        auto& C = ThreadLocalCleaner::thread_c;
        std::unique_ptr<FILE,decltype(&std::fclose)>
            fp(std::fopen(file.c_str(),"rb"),&std::fclose);
        if(!fp) throw std::runtime_error("open "+file);
        const Header h = readHeader(fp.get(),file);

        const uint64_t bx=x1-x0+1, by=y1-y0+1, bz=z1-z0+1;
        const uint64_t vox = bx*by*bz;
        if(U.size()<vox) U.resize(vox);

        char* dst = reinterpret_cast<char*>(U.data());
        uint64_t off=0;
        for(uint32_t c=0;c<h.nChunks;++c){
            if(C.size()<h.cLen[c]) C.resize(h.cLen[c]);
            freadExact(fp.get(),C.data(),h.cLen[c],"chunk");
            int dec=LZ4_decompress_safe(C.data(), dst+off, int(h.cLen[c]), int(h.uLen[c]));
            if(dec<0||uint64_t(dec)!=h.uLen[c]) throw std::runtime_error("LZ4");
            off+=h.uLen[c];
        }
        if(off!=h.total) throw std::runtime_error("size mismatch");

        /* 2. single-precision constants --------------------------------- */
        const bool useDmin = (!clip && dmin>0.f);
        const float clipSpan = high_clip-low_clip;
        const float scaleClip = clip ? scal*ampl/clipSpan : 0.f;
        const float scaleNC0 = scal*ampl/dmax;
        const float scaleNC1 = scal*ampl/(dmax-dmin);

        const float* src=U.data();

        /* 3. per-voxel loop --------------------------------------------- */
#if defined(__GNUC__)||defined(__clang__)
#pragma GCC ivdep
#pragma GCC unroll 8
#endif
        for(uint64_t z=0;z<bz;++z)
        for(uint64_t y=0;y<by;++y){
            const uint64_t base = idx3D(x0,y0+y,z0+z,dimX,dimY);
            for(uint64_t x=0;x<bx;++x){
                float R = src[(z*by+y)*bx+x];

                if(clip){
                    R = fmaf(R,1.f,-low_clip);
                    R = (R<0.f)?0.f:((R>clipSpan)?clipSpan:R);
                    R = fmaf(R,scaleClip,0.f);
                }else if(useDmin){
                    R = fmaf(R,1.f,-dmin);
                    R = fmaf(R,scaleNC1,0.f);
                }else{
                    R = fmaf(R,scaleNC0,0.f);
                }

                R -= ampl;
                R  = (R>=0.f) ? floorf(R+0.5f) : ceilf(R-0.5f);
                R  = (R <0.f) ? 0.f            : ( (R > scal) ? scal : R );

                vol[base+x]=R;
            }
        }
    }
};

/* ── MEX entry (unchanged plumbing) ─────────────────────────────────── */
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
{
    const auto tStart = std::chrono::high_resolution_clock::now();

    if(nrhs<12) mexErrMsgTxt("load_slab_lz4: wrong arg count");

    if(!mxIsCell(prhs[0])) mexErrMsgTxt("fnames must be cellstr");
    const mwSize N = mxGetNumberOfElements(prhs[0]);

    /* volume dims ------------------------------------------------------- */
    uint64_t dimX,dimY,dimZ;
    if(mxIsUint64(prhs[3])){
        auto p=reinterpret_cast<uint64_t*>(mxGetData(prhs[3]));
        dimX=p[0]; dimY=p[1]; dimZ=p[2];
    }else{
        auto p=mxGetPr(prhs[3]);
        dimX=p[0]; dimY=p[1]; dimZ=p[2];
    }

    /* scalar params (double -> float once) ------------------------------ */
    const bool   clipOn = (mxGetScalar(prhs[4])>0.0);
    const float  scal   = (float)mxGetScalar(prhs[5]);
    const float  ampl   = (float)mxGetScalar(prhs[6]);
    const float  dmin   = (float)mxGetScalar(prhs[7]);
    const float  dmax   = (float)mxGetScalar(prhs[8]);
    const float  lowC   = (float)mxGetScalar(prhs[9]);
    const float  highC  = (float)mxGetScalar(prhs[10]);
    int maxT = (nrhs>11)?(int)mxGetScalar(prhs[11])
                        :(int)std::thread::hardware_concurrency();
    if(maxT<1) maxT=1;

    /* output ------------------------------------------------------------ */
    mwSize md[3]={(mwSize)dimX,(mwSize)dimY,(mwSize)dimZ};
    mxClassID cls=(scal<=255)?mxUINT8_CLASS:mxUINT16_CLASS;
    mxArray* out=mxCreateNumericArray(3,md,cls,mxREAL);
    float* work=(float*)mxMalloc(sizeof(float)*dimX*dimY*dimZ);

    /* build all jobs ---------------------------------------------------- */
    const mxArray* p1=prhs[1]; const mxArray* p2=prhs[2];
    auto idx=[&](const mxArray* A,mwSize lin)->uint64_t{
        return mxIsUint64(A)?((uint64_t*)mxGetData(A))[lin]
                            :(uint64_t)mxGetPr(A)[lin];
    };
    std::vector<BrickJob> jobs; jobs.reserve(N);
    for(mwSize i=0;i<N;++i){
        char* cstr=mxArrayToUTF8String(mxGetCell(prhs[0],i));
        std::string fn(cstr); mxFree(cstr);

        uint64_t x0=idx(p1,i)-1, y0=idx(p1,i+N)-1,   z0=idx(p1,i+2*N)-1;
        uint64_t x1=idx(p2,i)-1, y1=idx(p2,i+N)-1,   z1=idx(p2,i+2*N)-1;

        jobs.push_back({fn,x0,y0,z0,x1,y1,z1,dimX,dimY,dimZ,work,
                        scal,ampl,lowC,highC,dmin,dmax,clipOn});
    }

    /* threaded execution ----------------------------------------------- */
    try{
        ThreadPool pool(maxT);
        for (const auto& j : jobs) pool.enqueue([job = j]{ job(); });
        pool.wait();
    }catch(const std::exception& e){
        mxFree(work);
        mexErrMsgTxt(e.what());
    }

    /* convert float work → uint{8,16} ---------------------------------- */
    if(cls==mxUINT8_CLASS){
        auto* dst=(uint8_t*)mxGetData(out);
        for(uint64_t i=0;i<dimX*dimY*dimZ;++i) dst[i]=(uint8_t)work[i];
    }else{
        auto* dst=(uint16_t*)mxGetData(out);
        for(uint64_t i=0;i<dimX*dimY*dimZ;++i) dst[i]=(uint16_t)work[i];
    }
    mxFree(work);
    plhs[0]=out;
    if (nlhs > 1)
        plhs[1] = mxCreateDoubleScalar(
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count());
}
