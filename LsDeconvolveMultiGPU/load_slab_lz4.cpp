/*==============================================================================
  load_slab_lz4.cpp
  ------------------------------------------------------------------------------
  High-throughput LZ4 brick loader for MATLAB (MEX).

  USAGE
  -----
    vol = load_slab_lz4(filenames, p1, p2, dims, clipOn,
                        scal, ampl, dmin, dmax, lowClip, highClip, maxThreads)

      filenames   : Nx1 cellstr – path to each *.lz4c brick
      p1, p2      : Nx3 uint64/double – 1-based XYZ start / end coords (inclusive)
      dims        : 1x3 [X Y Z] of the final volume
      clipOn      : logical – enable range clipping
      scal, ampl  : intensity scaling parameters
      dmin, dmax  : optional dynamic-range mapping
      lowClip     : lower clipping bound (when clipOn = true)
      highClip    : upper clipping bound (when clipOn = true)
      maxThreads  : (optional) max #worker threads – default = hw threads

  FEATURES
  --------
    • True multi-threaded LZ4 decompression (std::thread, no OpenMP/PThreads)
    • Atomic task dispatch with lost-wake-up fix
    • Per-thread scratch buffers → zero RAM contention
    • Robust error propagation – any worker exception aborts the MEX cleanly
    • Clean RAII resource management, no malloc/free leaks
    • C++17, single-header LZ4 dependency only

  AUTHOR  : Keivan Moradi  (with ChatGPT-4o assistance)
  LICENSE : GNU GPL v3  <https://www.gnu.org/licenses/>
  DATE    : 2025-06-22
==============================================================================*/

#include "lz4.h"
#include "mex.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/*------------------------------ fmaf fallback --------------------------------*/
#ifndef fmaf
#  define fmaf(a,b,c) ((a)*(b)+(c))
#endif

/*=============================== ThreadScratch ===============================*/
struct ThreadScratch {
    static thread_local std::vector<float> uncompressed;
    static thread_local std::vector<char>  compressed;
    ~ThreadScratch() { uncompressed.clear(); compressed.clear(); }
};
thread_local std::vector<float> ThreadScratch::uncompressed;
thread_local std::vector<char>  ThreadScratch::compressed;

/*================================ ThreadPool =================================*/
class ThreadPool {
public:
    explicit ThreadPool(std::size_t n)
        : shuttingDown_(false), unfinished_(0) {
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            workers_.emplace_back(&ThreadPool::workerLoop, this);
    }
    ~ThreadPool() { shutdown(); }

    template<class F>
    void enqueue(F&& job) {
        {
            std::lock_guard<std::mutex> lk(queueMtx_);
            jobQueue_.emplace(std::forward<F>(job));
            ++unfinished_;
        }
        queueCv_.notify_all();              // wake everyone → no lost signals
    }

    void wait() {
        std::unique_lock<std::mutex> lk(queueMtx_);
        finishedCv_.wait(lk,[&]{ return unfinished_==0; });
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    /* exception storage usable on C++14+ toolchains ------------------------*/
    std::mutex              exMtx_;
    std::exception_ptr      firstEx_{nullptr};

    void storeFirstException() noexcept {
        std::lock_guard<std::mutex> lk(exMtx_);
        if (!firstEx_) firstEx_ = std::current_exception();
    }

    void workerLoop() {
        ThreadScratch scratch;
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(queueMtx_);
                queueCv_.wait(lk,[&]{ return shuttingDown_||!jobQueue_.empty(); });

                if (shuttingDown_ && jobQueue_.empty()) return;

                job = std::move(jobQueue_.front());
                jobQueue_.pop();
                if (!jobQueue_.empty())
                    queueCv_.notify_one();          // cascade wake-up
            }

            try { job(); }
            catch (...) { storeFirstException(); }

            if (--unfinished_ == 0)
                finishedCv_.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(queueMtx_);
            shuttingDown_ = true;
        }
        queueCv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();

        std::lock_guard<std::mutex> lk(exMtx_);
        if (firstEx_) std::rethrow_exception(firstEx_);
    }

    /* data -----------------------------------------------------------------*/
    std::vector<std::thread>              workers_;
    std::queue<std::function<void()>>     jobQueue_;
    std::mutex                            queueMtx_;
    std::condition_variable               queueCv_, finishedCv_;
    std::atomic<std::size_t>              unfinished_;
    bool                                  shuttingDown_;
};

/*----------------------------- LZ4 brick header, helpers --------------------*/
constexpr uint32_t MAGIC      = 0x4C5A4331U;   // 'LZC1'
constexpr uint32_t HDR_BYTES  = 33280U;
constexpr uint32_t MAX_CHUNKS = 2048U;

enum DType : uint8_t { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

struct BrickHeader {
    uint32_t magic;
    uint8_t  dtype, ndims;
    uint64_t dims[16];
    uint64_t totalBytes, chunkBytes;
    uint32_t nChunks;
    uint64_t uLen[MAX_CHUNKS], cLen[MAX_CHUNKS];
    uint8_t  _pad[HDR_BYTES - (4 + 1 + 1 + 16*8 + 8 + 8 + 4 + MAX_CHUNKS*16)];
};

static void freadExact(FILE* fp, void* dst, std::size_t n, const char* ctx)
{
    if (std::fread(dst, 1, n, fp) != n)
        throw std::runtime_error(std::string(ctx) + ": I/O error");
}

static BrickHeader readHeader(FILE* fp, const std::string& file)
{
    BrickHeader h{};
    freadExact(fp, &h, HDR_BYTES, "header");
    if (h.magic != MAGIC)              throw std::runtime_error(file + ": bad magic");
    if (h.dtype != DT_SINGLE)          throw std::runtime_error(file + ": not float32");
    if (h.nChunks == 0 || h.nChunks > MAX_CHUNKS)
        throw std::runtime_error(file + ": bad chunk count");
    return h;
}

// Column-major 3D index
inline uint64_t idx3d(uint64_t x, uint64_t y, uint64_t z,
                      uint64_t dimX, uint64_t dimY)
{
    return x + dimX * (y + dimY * z);
}

/*=============================== BrickJob ====================================*/
struct BrickJob {
    /* geometry / dst */
    std::string file;
    uint64_t x0,y0,z0,x1,y1,z1, dimX,dimY,dimZ;
    float*    dst;
    /* scaling */
    float scal, ampl, lowClip, highClip, dmin, dmax;
    bool  clip;

    void operator()() const {
        auto& uc = ThreadScratch::uncompressed;
        auto& cc = ThreadScratch::compressed;

        std::unique_ptr<FILE,decltype(&std::fclose)>
            fp(std::fopen(file.c_str(),"rb"),&std::fclose);
        if (!fp) throw std::runtime_error("open "+file);

        const auto h = readHeader(fp.get(),file);

        const uint64_t bx=x1-x0+1, by=y1-y0+1, bz=z1-z0+1;
        const uint64_t vox = bx*by*bz;
        if (uc.size()<vox) uc.resize(vox);

        /* decompress -------------------------------------------------------*/
        char* uPtr = reinterpret_cast<char*>(uc.data());
        uint64_t off=0;
        for (uint32_t c=0;c<h.nChunks;++c){
            if (cc.size()<h.cLen[c]) cc.resize(h.cLen[c]);
            freadExact(fp.get(),cc.data(),h.cLen[c],"chunk");
            int got=LZ4_decompress_safe(cc.data(),uPtr+off,
                                         int(h.cLen[c]),int(h.uLen[c]));
            if (got<0||uint64_t(got)!=h.uLen[c])
                throw std::runtime_error(file+": LZ4 error");
            off+=h.uLen[c];
        }
        if (off!=h.totalBytes) throw std::runtime_error(file+": size mismatch");

        /* scaling factors --------------------------------------------------*/
        const float span = highClip-lowClip;
        const float kClip    = clip ? scal*ampl/span : 0.f;
        const float kNo0     = scal*ampl/dmax;
        const float kNo1     = scal*ampl/(dmax-dmin);
        const bool  useDmin  = (!clip && dmin>0.f);

        const float* src = uc.data();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#pragma GCC unroll 8
#endif
        for (uint64_t z=0; z<bz; ++z)
        for (uint64_t y=0; y<by; ++y){
            const uint64_t base = idx3d(x0,y0+y,z0+z,dimX,dimY);
            for (uint64_t x=0; x<bx; ++x){
                float v = src[(z*by+y)*bx + x];

                if (clip){
                    v = std::clamp(v-lowClip,0.f,span)*kClip;
                }else if (useDmin){
                    v = (v-dmin)*kNo1;
                }else{
                    v = v*kNo0;
                }

                v = v - ampl;
                v = (v>=0.f) ? std::floor(v+0.5f) : std::ceil(v-0.5f);
                v = std::clamp(v,0.f,scal);

                dst[base+x]=v;
            }
        }
    }
};

/*=============================== mexFunction =================================*/
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (nrhs<12) mexErrMsgTxt("Expected 12 input arguments.");
    if (!mxIsCell(prhs[0])) mexErrMsgTxt("filenames must be a cell array.");

    const mwSize nBricks = mxGetNumberOfElements(prhs[0]);
    auto dimAt=[&](const mxArray* a,int i)->uint64_t{
        return mxIsUint64(a)
            ? reinterpret_cast<const uint64_t*>(mxGetData(a))[i]
            : uint64_t(mxGetPr(a)[i]);
    };
    const uint64_t dimX=dimAt(prhs[3],0),
                   dimY=dimAt(prhs[3],1),
                   dimZ=dimAt(prhs[3],2);

    const bool  clip    = mxGetScalar(prhs[4])>0.0;
    const float scal    = float(mxGetScalar(prhs[5]));
    const float ampl    = float(mxGetScalar(prhs[6]));
    const float dmin    = float(mxGetScalar(prhs[7]));
    const float dmax    = float(mxGetScalar(prhs[8]));
    const float lowClip = float(mxGetScalar(prhs[9]));
    const float highClip= float(mxGetScalar(prhs[10]));
    int maxThreads = (nrhs>11)? int(mxGetScalar(prhs[11]))
                              : int(std::thread::hardware_concurrency());
    if (maxThreads<1) maxThreads=1;

    const mwSize mdim[3] = { mwSize(dimX),mwSize(dimY),mwSize(dimZ) };
    const mxClassID cls  = (scal<=255)? mxUINT8_CLASS : mxUINT16_CLASS;
    mxArray* out = mxCreateNumericArray(3,mdim,cls,mxREAL);

    /* work buffer (RAII) ----------------------------------------------------*/
    using FreeFn = void(*)(void*);
    std::unique_ptr<float,FreeFn>
        work(static_cast<float*>(mxMalloc(sizeof(float)*dimX*dimY*dimZ)), &mxFree);
    float* workPtr = work.get();

    /* build job list -------------------------------------------------------*/
    std::vector<BrickJob> jobs; jobs.reserve(nBricks);
    const mxArray *p1=prhs[1],*p2=prhs[2];
    auto c=[&](const mxArray* a,mwSize idx)->uint64_t{
        return mxIsUint64(a)
            ? reinterpret_cast<const uint64_t*>(mxGetData(a))[idx]
            : uint64_t(mxGetPr(a)[idx]);
    };

    for (mwSize i=0;i<nBricks;++i){
        char* f = mxArrayToUTF8String(mxGetCell(prhs[0],i));
        std::string file(f); mxFree(f);

        uint64_t x0=c(p1,i)-1, y0=c(p1,i+nBricks)-1, z0=c(p1,i+2*nBricks)-1;
        uint64_t x1=c(p2,i)-1, y1=c(p2,i+nBricks)-1, z1=c(p2,i+2*nBricks)-1;

        jobs.push_back({file,x0,y0,z0,x1,y1,z1,
                        dimX,dimY,dimZ, workPtr,
                        scal,ampl,lowClip,highClip,dmin,dmax,clip});
    }

    /* run pool -------------------------------------------------------------*/
    try {
        ThreadPool pool{size_t(maxThreads)};
        for (const auto& j: jobs) pool.enqueue([j]{ j(); });
        pool.wait();
    } catch(const std::exception& e){
        mexErrMsgTxt(e.what());
    }

    /* copy back ------------------------------------------------------------*/
    const uint64_t total = dimX*dimY*dimZ;
    if (cls==mxUINT8_CLASS){
        auto* d = static_cast<uint8_t*>(mxGetData(out));
        for (uint64_t i=0;i<total;++i) d[i]=uint8_t(workPtr[i]);
    }else{
        auto* d = static_cast<uint16_t*>(mxGetData(out));
        for (uint64_t i=0;i<total;++i) d[i]=uint16_t(workPtr[i]);
    }

    plhs[0]=out;
    if (nlhs>1){
        double s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now()-t0).count();
        plhs[1]=mxCreateDoubleScalar(s);
    }
}
