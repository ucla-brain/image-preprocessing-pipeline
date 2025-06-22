/*==============================================================================
  load_slab_lz4.cpp
  ------------------------------------------------------------------------------
  High-throughput LZ4 brick loader for MATLAB (MEX), using lock-free atomic
  dispatch (no mutex/condvar), zero-leak RAII, per-thread scratch, robust
  error handling.

  USAGE
  -----
    vol = load_slab_lz4(filenames, p1, p2, dims, clipOn,
                        scal, ampl, dmin, dmax, lowClip, highClip, maxThreads)
      See details below.

  FEATURES
  --------
    • True multi-threaded LZ4 decompression (std::thread)
    • **Lock-free atomic dispatch** (no mutex/condvar)
    • Per-thread scratch buffer, robust error propagation
    • Early casting to uint8/uint16 to save memory
    • Clean RAII: zero leaks, all exceptions safe
    • C++17, single-header LZ4 dependency only

  LIMITATIONS
  -----------
    • Only float32 bricks supported (see DT_SINGLE)
    • No progress reporting/cancellation
    • No support for MATLAB complex/NaN values
    • maxThreads > N_jobs is OK (workers exit if no more jobs)

  AUTHOR  : Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE : GNU GPL v3 <https://www.gnu.org/licenses/>
  DATE    : 2025-06-22
==============================================================================*/

#include "lz4.h"
#include "mex.h"

#include <atomic>
#include <cstdio>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>

/*------------------------------ fmaf fallback --------------------------------*/
#ifndef fmaf
#  define fmaf(a,b,c) ((a)*(b)+(c))
#endif

/*========================== Thread-local Scratch Buffers =====================*/
template<typename T>
struct ThreadScratch {
    static thread_local std::vector<T> uncompressed;
    static thread_local std::vector<char> compressed;
    ~ThreadScratch() { uncompressed.clear(); compressed.clear(); }
};
template<typename T>
thread_local std::vector<T> ThreadScratch<T>::uncompressed;
template<typename T>
thread_local std::vector<char> ThreadScratch<T>::compressed;

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

// Column-major 3D index (MATLAB order)
inline uint64_t idx3d(uint64_t x, uint64_t y, uint64_t z,
                      uint64_t dimX, uint64_t dimY)
{
    return x + dimX * (y + dimY * z);
}

/*=============================== BrickJob ====================================*/
// Each job decompresses a brick and copies/casts it into the shared output.
template<typename OUT_T>
struct BrickJob {
    std::string file;
    uint64_t x0,y0,z0,x1,y1,z1, dimX,dimY,dimZ;
    OUT_T*   dst;
    float    scal, ampl, lowClip, highClip, dmin, dmax;
    bool     clip;

    void operator()() const {
        auto& uc = ThreadScratch<float>::uncompressed;
        auto& cc = ThreadScratch<float>::compressed;

        // Open file and parse header
        std::unique_ptr<FILE,decltype(&std::fclose)>
            fp(std::fopen(file.c_str(),"rb"), &std::fclose);
        if (!fp) throw std::runtime_error("open "+file);

        const auto h = readHeader(fp.get(),file);

        const uint64_t bx=x1-x0+1, by=y1-y0+1, bz=z1-z0+1;
        const uint64_t vox = bx*by*bz;
        if (uc.size()<vox) uc.resize(vox);

        // Decompress all chunks
        char* uPtr = reinterpret_cast<char*>(uc.data());
        uint64_t off=0;
        for (uint32_t c=0;c<h.nChunks;++c){
            if (cc.size()<h.cLen[c]) cc.resize(h.cLen[c]);
            freadExact(fp.get(),cc.data(),h.cLen[c],"chunk");
            int got = LZ4_decompress_safe(cc.data(), uPtr+off, int(h.cLen[c]), int(h.uLen[c]));
            if (got<0 || uint64_t(got)!=h.uLen[c])
                throw std::runtime_error(file+": LZ4 error");
            off += h.uLen[c];
        }
        if (off!=h.totalBytes) throw std::runtime_error(file+": size mismatch");

        // Calculate mapping factors (same as original)
        const float span = highClip-lowClip;
        const float kClip    = clip ? scal*ampl/span : 0.f;
        const float kNo0     = scal*ampl/dmax;
        const float kNo1     = scal*ampl/(dmax-dmin);
        const bool  useDmin  = (!clip && dmin>0.f);

        const float* src = uc.data();

        // Early cast to OUT_T as we write into final volume
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

                // Early cast
                dst[base+x] = OUT_T(v);
            }
        }
    }
};

/*========================= Atomic ThreadPool (No Condition Variables) ========*/
template<typename JobT>
void run_atomic_thread_pool(const std::vector<JobT>& jobs, int nThreads)
{
    // Atomic index for job dispatch
    std::atomic<size_t> nextJob{0};
    std::atomic<bool> exceptionHappened{false};
    std::string exceptionMsg;

    auto worker = [&]() {
        while (true) {
            // Get next job index
            size_t idx = nextJob.fetch_add(1, std::memory_order_relaxed);
            if (idx >= jobs.size() || exceptionHappened.load(std::memory_order_relaxed))
                break;

            try {
                jobs[idx]();
            } catch (const std::exception& ex) {
                // Only the first exception is recorded; all workers abort soon
                if (!exceptionHappened.exchange(true))
                    exceptionMsg = ex.what();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (int t=0; t<nThreads; ++t)
        threads.emplace_back(worker);

    for (auto& th : threads)
        th.join();

    if (exceptionHappened && !exceptionMsg.empty())
        throw std::runtime_error(exceptionMsg);
}

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

    mxArray* out = mxCreateNumericArray(3, mdim,cls,mxREAL);

    // Output pointer for final data
    void* volPtr = mxGetData(out);

    // Build job list
    const mxArray *p1=prhs[1],*p2=prhs[2];
    auto c=[&](const mxArray* a,mwSize idx)->uint64_t{
        return mxIsUint64(a)
            ? reinterpret_cast<const uint64_t*>(mxGetData(a))[idx]
            : uint64_t(mxGetPr(a)[idx]);
    };

    if (cls == mxUINT8_CLASS) {
        std::vector<BrickJob<uint8_t>> jobs; jobs.reserve(nBricks);
        for (mwSize i=0;i<nBricks;++i){
            char* f = mxArrayToUTF8String(mxGetCell(prhs[0],i));
            std::string file(f); mxFree(f);

            uint64_t x0=c(p1,i)-1, y0=c(p1,i+nBricks)-1, z0=c(p1,i+2*nBricks)-1;
            uint64_t x1=c(p2,i)-1, y1=c(p2,i+nBricks)-1, z1=c(p2,i+2*nBricks)-1;

            jobs.push_back({file,x0,y0,z0,x1,y1,z1,
                            dimX,dimY,dimZ, static_cast<uint8_t*>(volPtr),
                            scal,ampl,lowClip,highClip,dmin,dmax,clip});
        }
        try {
            run_atomic_thread_pool(jobs, maxThreads);
        } catch(const std::exception& e){
            mexErrMsgTxt(e.what());
        }
    } else {
        std::vector<BrickJob<uint16_t>> jobs; jobs.reserve(nBricks);
        for (mwSize i=0;i<nBricks;++i){
            char* f = mxArrayToUTF8String(mxGetCell(prhs[0],i));
            std::string file(f); mxFree(f);

            uint64_t x0=c(p1,i)-1, y0=c(p1,i+nBricks)-1, z0=c(p1,i+2*nBricks)-1;
            uint64_t x1=c(p2,i)-1, y1=c(p2,i+nBricks)-1, z1=c(p2,i+2*nBricks)-1;

            jobs.push_back({file,x0,y0,z0,x1,y1,z1,
                            dimX,dimY,dimZ, static_cast<uint16_t*>(volPtr),
                            scal,ampl,lowClip,highClip,dmin,dmax,clip});
        }
        try {
            run_atomic_thread_pool(jobs, maxThreads);
        } catch(const std::exception& e){
            mexErrMsgTxt(e.what());
        }
    }

    plhs[0]=out;
    if (nlhs>1){
        double s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now()-t0).count();
        plhs[1]=mxCreateDoubleScalar(s);
    }
}
