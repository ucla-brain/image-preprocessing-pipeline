/*==============================================================================
  load_blocks_lz4_mex.cpp
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
             load_blocks_lz4_mex.cpp lz4.c

  AUTHORSHIP
      Initial specification : Keivan Moradi
      Implementation        : ChatGPT-O3 and 4o  (June 2025)

  LICENSE
      GNU General Public License v3.0  –  https://www.gnu.org/licenses/gpl-3.0
==============================================================================*/

// BEGIN: Optimized load_blocks_lz4_mex.cpp (Performance Focus)
// NOTE: Keep LZ4, header parsing, and idx3D definitions the same as before...

#include "mex.h"
#include "matrix.h"
#include "lz4.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <memory>
#include <functional>   // << REQUIRED

/*==============================================================================
 *                       1.  Simple C++17 Thread Pool
 *============================================================================*/
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i)
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {   // Lock scope
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
    }

    template<class F>
    void enqueue(F&& f) {
        {   // Lock scope
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queueMutex);
        condition.wait(lock, [this] {
            return tasks.empty();
        });
    }

    ~ThreadPool() {
        {   // Lock scope
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

/*==============================================================================
 *                   2.  BrickJob (Rewritten for Buffer Reuse)
 *============================================================================*/
struct BrickJob {
    std::string file;
    uint64_t x0, y0, z0, x1, y1, z1;
    uint64_t dimX, dimY, dimZ;
    float* volPtr;

    void operator()() const {
        static thread_local std::vector<float> uBuffer;
        static thread_local std::vector<char> cBuf;

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
        uBuffer.resize(totalVoxels);
        char* dst = reinterpret_cast<char*>(uBuffer.data());

        uint64_t offset = 0;
        for (uint32_t c = 0; c < h.numChunks; ++c) {
            const uint64_t compB = h.chunkComp[c];
            const uint64_t uncomp = h.chunkUncomp[c];

            if (compB > 0x7FFFFFFF || uncomp > 0x7FFFFFFF)
                throw std::runtime_error(file + ": chunk > 2 GB");

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
 *                  3.  MEX Entry – Uses ThreadPool for Dispatch
 *============================================================================*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    const auto tStart = std::chrono::high_resolution_clock::now();
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgTxt("Usage: [vol, elapsed] = load_blocks_lz4_mex(fnames, p1, p2, volSize, [maxThreads])");

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
        std::atomic<size_t> completed{0};

        for (const auto& job : jobs)
            pool.enqueue([&job, &completed]() {
                job();
                ++completed;
            });

        while (completed.load() < jobs.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("load_blocks_lz4_mex:ThreadError", e.what());
    }

    plhs[0] = volMx;
    if (nlhs > 1)
        plhs[1] = mxCreateDoubleScalar(
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count());
}
