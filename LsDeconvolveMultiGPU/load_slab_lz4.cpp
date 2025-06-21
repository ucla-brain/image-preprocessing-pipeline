/*==============================================================================
  load_slab_lz4.cpp
  ------------------------------------------------------------------------------
  Reconstruct a very large 3-D single-precision volume from many *.lz4 bricks,
  loading each brick in parallel C++ threads and writing directly into a
  shared MATLAB array (no MATLAB-side workers required).

  ──────────────────────────────────────────────────────────────────────────────
  HOW IT WORKS
  • MATLAB passes:
        - cellstr   brickFiles   : one *.lz4c per brick   (N×1)
        - uint/dbl  brickP1, P2  : [N×3] 1-based XYZ mins / maxes
        - uint/dbl  volSize      : [1×3] final volume dims
        - int       maxThreads   : optional thread cap
  • MEX allocates the output array once (single precision, size = volSize).
  • A lightweight thread pool (std::async) processes bricks:
        1.  Read custom LZ4 header  (32 KiB)
        2.  Decompress all chunks   → per-thread std::vector<float>
        3.  memcpy row-by-row into the correct location in the shared volume.
  • All writes are non-overlapping → no mutexes needed.
  • On any error, an exception bubbles up; MATLAB receives a precise message.

  COMPILATION
      >> mex -R2018a CXXFLAGS="$CXXFLAGS -std=c++17 -O3" ...
             load_slab_lz4.cpp lz4.c

  AUTHORSHIP
      Initial specification : Keivan Moradi
      Implementation        : ChatGPT-O3 and 4o  (June 2025)

  LICENSE
      GNU General Public License v3.0  –  https://www.gnu.org/licenses/gpl-3.0
==============================================================================*/

#include "lz4.h"
#include "matrix.h"
#include "mex.h"

#include <atomic>
#include <chrono>
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

/*==============================================================================
 *                         Thread-Local Cleanup Helper
 *============================================================================*/
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

/*==============================================================================
 *                           Thread Pool Implementation
  AtomicThreadPool
  ----------------------------------------------------------
  • FIFO task queue protected by a mutex
  • Atomic counter tracks tasks still “in flight”
  • Two condition variables:
        – cv_job_   : workers sleep here waiting for work
        – cv_done_  : the main thread sleeps here in wait()
  • When the counter hits zero the last worker notifies cv_done_
 *============================================================================*/
class ThreadPool
{
public:
    explicit ThreadPool(std::size_t numThreads)
        : stop_{false}, pending_{0}
    {
        for (std::size_t i = 0; i < numThreads; ++i)
            workers_.emplace_back([this]
            {
                ThreadLocalCleaner cleaner;           // per-thread buffer cleanup
                while (true)
                {
                    std::function<void()> task;
                    {   // ---- get next job (or quit) ----
                        std::unique_lock<std::mutex> lk(queueMtx_);
                        cv_job_.wait(lk, [this]
                        {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    // ---- run job ----
                    task();

                    // ---- one task done ----
                    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    {
                        std::lock_guard<std::mutex> lk(doneMtx_);
                        cv_done_.notify_one();          // last task woke the waiter
                    }
                }
            });
    }

    template<class F>
    void enqueue(F&& f)
    {
        {
            std::lock_guard<std::mutex> lk(queueMtx_);
            if (stop_)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace(std::forward<F>(f));
            pending_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_job_.notify_one();
    }

    /*------------------------------------------------------------------
        Blocks the caller until **all** enqueued tasks have finished.
        Safe to call multiple times; returns immediately if idle.
    ------------------------------------------------------------------*/
    void wait()
    {
        std::unique_lock<std::mutex> lk(doneMtx_);
        cv_done_.wait(lk, [this] { return pending_.load(std::memory_order_acquire) == 0; });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lk(queueMtx_);
            stop_ = true;
        }
        cv_job_.notify_all();              // wake every sleeper
        for (auto& t : workers_) t.join(); // join in order
    }

private:
    /*--- worker side ---*/
    std::vector<std::thread>            workers_;
    std::queue<std::function<void()>>   tasks_;
    std::mutex                          queueMtx_;
    std::condition_variable             cv_job_;

    /*--- completion side ---*/
    std::atomic<std::size_t>            pending_;
    std::mutex                          doneMtx_;
    std::condition_variable             cv_done_;

    /*--- state ---*/
    bool                                stop_;
};

/*==============================================================================
 *                     File-format constants & helpers
 *============================================================================*/
constexpr uint32_t MAGIC_NUMBER = 0x4C5A4331U;   // 'LZ4C'
constexpr uint32_t HEADER_SIZE  = 33280U;        // bytes (fixed)
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

static void freadExact(FILE* f, void* dst, size_t n, const char* context)
{
    if (std::fread(dst, 1, n, f) != n)
        throw std::runtime_error(std::string("I/O error while ") + context);
}

static FileHeader readHeader(FILE* f, const std::string& fname)
{
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
                             uint64_t dimX, uint64_t dimY)
{
    return x + dimX * (y + dimY * z);
}

/*==============================================================================
 *                        BrickJob (Buffer Reuse Enabled)
 *============================================================================*/
struct BrickJob {
    std::string file;
    uint64_t x0, y0, z0, x1, y1, z1;
    uint64_t dimX, dimY, dimZ;
    float* volPtr;

    void operator()() const {
        auto& uBuffer = ThreadLocalCleaner::thread_local_uBuffer;
        auto& cBuf    = ThreadLocalCleaner::thread_local_cBuf;

        std::unique_ptr<FILE, decltype(&std::fclose)>
            fp(std::fopen(file.c_str(), "rb"), &std::fclose);
        if (!fp)
            throw std::runtime_error("Cannot open file: " + file);

        const FileHeader h = readHeader(fp.get(), file);

        const uint64_t brickX = x1 - x0 + 1;
        const uint64_t brickY = y1 - y0 + 1;
        const uint64_t brickZ = z1 - z0 + 1;

        if (brickX != h.dims[0] || brickY != h.dims[1] || brickZ != ((h.ndims == 3) ? h.dims[2] : 1))
            throw std::runtime_error(file + ": dims in header ≠ expected brick dims");

        const uint64_t totalVoxels = brickX * brickY * brickZ;
        if (uBuffer.size() < totalVoxels)
            uBuffer.resize(totalVoxels);

        char* dst = reinterpret_cast<char*>(uBuffer.data());
        uint64_t offset = 0;

        for (uint32_t c = 0; c < h.numChunks; ++c) {
            const uint64_t compB = h.chunkComp[c];
            const uint64_t uncomp = h.chunkUncomp[c];

            if (compB > 0x7FFFFFFF || uncomp > 0x7FFFFFFF)
                throw std::runtime_error(file + ": chunk > 2 GB");

            if (cBuf.size() < compB)
                cBuf.resize(compB);

            freadExact(fp.get(), cBuf.data(), compB, ("reading chunk of " + file).c_str());

            const int decoded = LZ4_decompress_safe(cBuf.data(), dst + offset,
                                                    static_cast<int>(compB),
                                                    static_cast<int>(uncomp));
            if (decoded < 0 || static_cast<uint64_t>(decoded) != uncomp)
                throw std::runtime_error(file + ": LZ4 decompression failed");

            offset += uncomp;
        }

        if (offset != h.totalUncompressed)
            throw std::runtime_error(file + ": size mismatch after decompress");

        const uint64_t rowElems = brickX;
        const float* src = uBuffer.data();
        for (uint64_t z = 0; z < brickZ; ++z)
            for (uint64_t y = 0; y < brickY; ++y) {
                const uint64_t dstIdx = idx3D(x0, y0 + y, z0 + z, dimX, dimY);
                std::memcpy(volPtr + dstIdx, src, rowElems * sizeof(float));
                src += rowElems;
            }
    }
};

/*==============================================================================
 *                               MEX Entry Point
 *============================================================================*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    const auto tStart = std::chrono::high_resolution_clock::now();
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgTxt("Usage: [vol, elapsed] = load_slabs_lz4(fnames, p1, p2, volSize, [maxThreads])");

    if (!mxIsCell(prhs[0])) mexErrMsgTxt("First arg must be cellstr");
    const mwSize N = mxGetNumberOfElements(prhs[0]);
    if (N == 0) mexErrMsgTxt("No files given");

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

    int maxThreads = (nrhs == 5) ? static_cast<int>(mxGetScalar(prhs[4]))
                                 : static_cast<int>(std::thread::hardware_concurrency());
    if (maxThreads < 1) maxThreads = 1;

    mwSize mdims[3] = { static_cast<mwSize>(dimX), static_cast<mwSize>(dimY), static_cast<mwSize>(dimZ) };
    mxArray* volMx = mxCreateNumericArray(3, mdims, mxSINGLE_CLASS, mxREAL);
    // mxArray* volMx = mxCreateUninitNumericArray(3, mdims, mxSINGLE_CLASS, mxREAL);
    if (!volMx) mexErrMsgTxt("Cannot allocate output volume");
    float* volPtr = static_cast<float*>(mxGetData(volMx));

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

        jobs.emplace_back(BrickJob{ fname, x0, y0, z0, x1, y1, z1, dimX, dimY, dimZ, volPtr });
    }

    try {
        ThreadPool pool(maxThreads);
        for (const auto& job : jobs)
            pool.enqueue([&job]() {
                job();
            });
        pool.wait();  // Waits until all jobs finish
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("load_slabs_lz4:ThreadError", e.what());
    }

    plhs[0] = volMx;
    if (nlhs > 1)
        plhs[1] = mxCreateDoubleScalar(
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count());
}
