/*==============================================================================
  load_slab_lz4.cpp
  -------------------------------------------------------------------------------
  Loads and processes a large 3D volume from LZ4 bricks, rescales/clips in threads,
  and outputs directly as uint8 or uint16 (RAM efficient).
==============================================================================*/

#include "lz4.h"
#include "matrix.h"
#include "mex.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

/*==============================================================================*/
struct ThreadLocalCleaner {
    ~ThreadLocalCleaner() {
        thread_local_uBuffer = std::vector<float>();
        thread_local_cBuf = std::vector<char>();
    }

    static thread_local std::vector<float> thread_local_uBuffer;
    static thread_local std::vector<char>  thread_local_cBuf;
};
thread_local std::vector<float> ThreadLocalCleaner::thread_local_uBuffer;
thread_local std::vector<char>  ThreadLocalCleaner::thread_local_cBuf;

/*==============================================================================*/
class ThreadPool {
public:
    explicit ThreadPool(std::size_t numThreads)
        : stop_{false}, pending_{0}
    {
        for (std::size_t i = 0; i < numThreads; ++i)
            workers_.emplace_back([this]
            {
                ThreadLocalCleaner cleaner;
                while (true) {
                    std::function<void()> task;
                    {   std::unique_lock<std::mutex> lk(queueMtx_);
                        cv_job_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front()); tasks_.pop();
                    }
                    task();
                    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(doneMtx_);
                        cv_done_.notify_one();
                    }
                }
            });
    }

    template<class F> void enqueue(F&& f) {
        {   std::lock_guard<std::mutex> lk(queueMtx_);
            if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace(std::forward<F>(f));
            pending_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_job_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(doneMtx_);
        cv_done_.wait(lk, [this] { return pending_.load(std::memory_order_acquire) == 0; });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(queueMtx_); stop_ = true; }
        cv_job_.notify_all();
        for (auto& t : workers_) t.join();
    }
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queueMtx_;
    std::condition_variable cv_job_;
    std::atomic<std::size_t> pending_;
    std::mutex doneMtx_;
    std::condition_variable cv_done_;
    bool stop_;
};

/*==============================================================================*/
constexpr uint32_t MAGIC_NUMBER = 0x4C5A4331U;
constexpr uint32_t HEADER_SIZE  = 33280U;
constexpr uint32_t MAX_DIMS     = 16;
constexpr uint32_t MAX_CHUNKS   = 2048;
enum DType : uint8_t { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

struct FileHeader {
    uint32_t magic;
    uint8_t  dtype;
    uint8_t  ndims;
    uint64_t dims[MAX_DIMS];
    uint64_t totalUncompressed;
    uint64_t chunkSize;
    uint32_t numChunks;
    uint64_t chunkUncomp[MAX_CHUNKS];
    uint64_t chunkComp  [MAX_CHUNKS];
    uint8_t  padding[HEADER_SIZE - (4 + 1 + 1 + 8*MAX_DIMS + 8 + 8 + 4 + 8*MAX_CHUNKS*2)];
};

static void freadExact(FILE* f, void* dst, size_t n, const char* context) {
    if (std::fread(dst, 1, n, f) != n)
        throw std::runtime_error(std::string("I/O error while ") + context);
}
static FileHeader readHeader(FILE* f, const std::string& fname) {
    FileHeader h{};
    freadExact(f, &h, HEADER_SIZE, ("reading header of " + fname).c_str());
    if (h.magic != MAGIC_NUMBER)
        throw std::runtime_error(fname + ": wrong magic number (not LZ4C)");
    if (h.dtype != DT_SINGLE)
        throw std::runtime_error(fname + ": unsupported dtype (expect single)");
    if (h.ndims != 2 && h.ndims != 3)
        throw std::runtime_error(fname + ": ndims must be 2 or 3");
    if (h.numChunks == 0 || h.numChunks > MAX_CHUNKS)
        throw std::runtime_error(fname + ": invalid chunk count");
    return h;
}
static inline uint64_t idx3D(uint64_t x, uint64_t y, uint64_t z,
                             uint64_t dimX, uint64_t dimY) {
    return x + dimX * (y + dimY * z);
}

/*==============================================================================
 *                               BrickJob
 *   – Thread-safe loader + scaler for one brick
 *   – All arithmetic in single precision
 *   – Uses FMA where available, pragma-hints for auto-vectorisation
 *==============================================================================*/
struct BrickJob {
    std::string file;
    uint64_t x0, y0, z0, x1, y1, z1;
    uint64_t dimX, dimY, dimZ;
    float*   volPtr;

    /* scalar parameters (double in ctor, cast once to float) */
    double scal_, amplification_, low_clip_, high_clip_, deconvmin_, deconvmax_;
    bool    clipOn;

    void operator()() const
    {
        /*---------------------------------- 1. open + decompress -------------*/
        auto& uBuf = ThreadLocalCleaner::thread_local_uBuffer;
        auto& cBuf = ThreadLocalCleaner::thread_local_cBuf;

        std::unique_ptr<FILE, decltype(&std::fclose)>
            fp(std::fopen(file.c_str(), "rb"), &std::fclose);
        if (!fp) throw std::runtime_error("Cannot open " + file);

        const FileHeader h = readHeader(fp.get(), file);
        const uint64_t brickX = x1 - x0 + 1,  brickY = y1 - y0 + 1,
                       brickZ = z1 - z0 + 1;

        if (brickX != h.dims[0] || brickY != h.dims[1] ||
            brickZ != ((h.ndims == 3) ? h.dims[2] : 1))
            throw std::runtime_error(file + ": dims mismatch");

        const uint64_t vox = brickX * brickY * brickZ;
        if (uBuf.size() < vox) uBuf.resize(vox);

        char* dst = reinterpret_cast<char*>(uBuf.data());
        uint64_t off = 0;
        for (uint32_t c = 0; c < h.numChunks; ++c) {
            if (cBuf.size() < h.chunkComp[c]) cBuf.resize(h.chunkComp[c]);
            freadExact(fp.get(), cBuf.data(), h.chunkComp[c], "chunk read");
            const int dec = LZ4_decompress_safe(cBuf.data(), dst + off,
                static_cast<int>(h.chunkComp[c]), static_cast<int>(h.chunkUncomp[c]));
            if (dec < 0 || static_cast<uint64_t>(dec) != h.chunkUncomp[c])
                throw std::runtime_error(file + ": LZ4 error");
            off += h.chunkUncomp[c];
        }
        if (off != h.totalUncompressed)
            throw std::runtime_error(file + ": size mismatch");

        /*---------------------------------- 2. pre-compute floats ------------*/
        const float scalF =  static_cast<float>(scal_);
        const float ampF  =  static_cast<float>(amplification_);
        const float lowF  =  static_cast<float>(low_clip_);
        const float highF =  static_cast<float>(high_clip_);
        const float dminF =  static_cast<float>(deconvmin_);
        const float dmaxF =  static_cast<float>(deconvmax_);

        const bool useDMin = (!clipOn && dminF > 0.f);

        const float clipSpanF = highF - lowF;
        if (clipOn && clipSpanF == 0.f)
            throw std::runtime_error("high_clip == low_clip (division by zero)");

        float scaleClip = 0.f;     // only valid if clipOn
        if (clipOn)
            scaleClip = static_cast<float>((double)scalF * ampF / (double)clipSpanF);

        const float scaleNC0 = static_cast<float>((double)scalF * ampF / (double)dmaxF);
        const float scaleNC1 = static_cast<float>((double)scalF * ampF /
                                                  ((double)dmaxF - (double)dminF));

        const float* src = uBuf.data();

        /*---------------------------------- 3. per-voxel loop ----------------*/
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#pragma GCC unroll 8
#endif
        for (uint64_t z = 0; z < brickZ; ++z)
        for (uint64_t y = 0; y < brickY; ++y)
        {
            const uint64_t base = idx3D(x0, y0 + y, z0 + z, dimX, dimY);
            for (uint64_t x = 0; x < brickX; ++x)
            {
                float v = src[(z * brickY + y) * brickX + x];

                /* rescale / clip */
                if (clipOn) {
                    v  = std::fmaf(v, 1.f, -lowF);                 // v -= lowF
                    v  = (v < 0.f) ? 0.f : (v > clipSpanF ? clipSpanF : v);
                    v  = std::fmaf(v, scaleClip, 0.f);             // v *= scaleClip
                } else if (useDMin) {
                    v  = std::fmaf(v, 1.f, -dminF);                // v -= dminF
                    v  = std::fmaf(v, scaleNC1, 0.f);              // v *= scaleNC1
                } else {
                    v  = std::fmaf(v, scaleNC0, 0.f);              // v *= scaleNC0
                }

                /* round half-away-from-zero & clamp */
                v -= ampF;
                v  = (v >= 0.f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
                v  = (v < 0.f) ? 0.f : (v > scalF ? scalF : v);

                volPtr[base + x] = v;   // store as float
            }
        }
    }
};


/*==============================================================================*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Argument order:
    // 0: fnames, 1: p1, 2: p2, 3: volSize, 4: clipval, 5: scal, 6: amplification,
    // 7: deconvmin, 8: deconvmax, 9: low_clip, 10: high_clip, 11: [maxThreads]
    const auto tStart = std::chrono::high_resolution_clock::now();

    if (nrhs < 12)
        mexErrMsgTxt("Usage: vol = load_slab_lz4(fnames, p1, p2, volSize, "
                     "clipval, scal, amplification, deconvmin, deconvmax, "
                     "low_clip, high_clip, [maxThreads])");

    // Argument unpacking
    // fnames : cellstr of brick files
    if (!mxIsCell(prhs[0])) mexErrMsgTxt("First arg must be cellstr");
    const mwSize N = mxGetNumberOfElements(prhs[0]);
    if (N == 0) mexErrMsgTxt("No files given");

    // Brick indices and volume size
    const mxArray* p1mx = prhs[1];
    const mxArray* p2mx = prhs[2];

    auto getIdx = [N](const mxArray* A, mwSize linear) -> uint64_t {
        return mxIsUint64(A) ? static_cast<uint64_t*>(mxGetData(A))[linear]
                             : static_cast<uint64_t>(mxGetPr(A)[linear]);
    };

    uint64_t dimX, dimY, dimZ;
    if (mxIsUint64(prhs[3])) {
        auto p = static_cast<uint64_t*>(mxGetData(prhs[3]));
        dimX = p[0]; dimY = p[1]; dimZ = p[2];
    } else {
        auto p = mxGetPr(prhs[3]);
        dimX = p[0]; dimY = p[1]; dimZ = p[2];
    }

    // All further parameters (scalars)
    bool clipOn         = (mxGetScalar(prhs[4]) > 0.0);
    double scal         = mxGetScalar(prhs[5]);
    double amplification= mxGetScalar(prhs[6]);
    double deconvmin    = mxGetScalar(prhs[7]);
    double deconvmax    = mxGetScalar(prhs[8]);
    double low_clip     = mxGetScalar(prhs[9]);
    double high_clip    = mxGetScalar(prhs[10]);
    int maxThreads      = (nrhs > 11) ? static_cast<int>(mxGetScalar(prhs[11]))
                                      : static_cast<int>(std::thread::hardware_concurrency());
    if (maxThreads < 1) maxThreads = 1;

    // Output type (uint8 or uint16)
    mxClassID outClass = (scal <= 255) ? mxUINT8_CLASS : mxUINT16_CLASS;

    // Allocate output
    mwSize mdims[3] = { static_cast<mwSize>(dimX), static_cast<mwSize>(dimY), static_cast<mwSize>(dimZ) };
    mxArray* volMx = mxCreateNumericArray(3, mdims, outClass, mxREAL);
    if (!volMx) mexErrMsgTxt("Cannot allocate output volume");
    float* volPtr = static_cast<float*>(mxMalloc(sizeof(float) * dimX * dimY * dimZ));
    if (!volPtr) mexErrMsgTxt("Cannot allocate working buffer");

    // Prepare jobs
    std::vector<BrickJob> jobs; jobs.reserve(N);
    for (mwSize i = 0; i < N; ++i) {
        char* tmp = mxArrayToUTF8String(mxGetCell(prhs[0], i));
        if (!tmp) mexErrMsgTxt("Invalid filename cell");
        std::string fname(tmp); mxFree(tmp);

        uint64_t x0 = getIdx(p1mx, i) - 1,
                 y0 = getIdx(p1mx, i + N) - 1,
                 z0 = getIdx(p1mx, i + 2 * N) - 1;
        uint64_t x1 = getIdx(p2mx, i) - 1,
                 y1 = getIdx(p2mx, i + N) - 1,
                 z1 = getIdx(p2mx, i + 2 * N) - 1;

        if (x1 < x0 || y1 < y0 || z1 < z0)
            mexErrMsgTxt("p1 > p2 for at least one brick");
        if (x1 >= dimX || y1 >= dimY || z1 >= dimZ)
            mexErrMsgTxt("Brick exceeds bounds");

        jobs.emplace_back(
            BrickJob{ fname, x0, y0, z0, x1, y1, z1, dimX, dimY, dimZ,
                      volPtr, scal, amplification, low_clip, high_clip,
                      deconvmin, deconvmax, clipOn }
        );
    }

    // Parallel load
    try {
        ThreadPool pool(maxThreads);
        for (const auto& job : jobs)
            pool.enqueue([&job]() { job(); });
        pool.wait();
    } catch (const std::exception& e) {
        mxFree(volPtr);
        mexErrMsgIdAndTxt("load_slab_lz4:ThreadError", e.what());
    }

    // Convert working buffer to uint8/uint16 output
    if (outClass == mxUINT8_CLASS) {
        uint8_t* outPtr = static_cast<uint8_t*>(mxGetData(volMx));
        for (mwSize i = 0; i < dimX * dimY * dimZ; ++i)
            outPtr[i] = static_cast<uint8_t>(volPtr[i]);
    } else {
        uint16_t* outPtr = static_cast<uint16_t*>(mxGetData(volMx));
        for (mwSize i = 0; i < dimX * dimY * dimZ; ++i)
            outPtr[i] = static_cast<uint16_t>(volPtr[i]);
    }
    mxFree(volPtr);

    plhs[0] = volMx;
    if (nlhs > 1)
        plhs[1] = mxCreateDoubleScalar(
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count());
}