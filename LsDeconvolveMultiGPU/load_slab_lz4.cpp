/*==============================================================================
  load_slab_lz4.cpp
  ------------------------------------------------------------------------------
  High-throughput LZ4 slab loader for MATLAB.

  USAGE
  -----
    vol = load_slab_lz4(filenames, p1, p2, dims, clipOn,
                        scal, ampl, dmin, dmax, lowClip, highClip, maxThreads)

  FEATURES
  --------
    • Parallel LZ4 decompression with per-thread reuse
    • Lock-free atomic dispatch (no mutex or condvar)
    • Early casting to reduce memory pressure (float→uint8/uint16)
    • Memory-safe, exception-safe (RAII)
    • Automatic thread reuse and buffer reuse
    • Robust error handling with descriptive messages
    • Clean separation of responsibilities

  LIMITATIONS
  -----------
    • Only float32 LZ4 bricks supported
    • No progress bar or cancellation
    • Assumes all header sizes are valid and consistent

  AUTHOR  : Keivan Moradi (with ChatGPT-4o assistance)
  LICENSE : GNU GPL v3
  DATE    : 2025-06-22
==============================================================================*/

#include "mex.h"
#include "lz4.h"

#include <atomic>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread>
#include <string>
#include <memory>
#include <stdexcept>
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
template<typename T> thread_local std::vector<T> ThreadScratch<T>::uncompressed;
template<typename T> thread_local std::vector<char> ThreadScratch<T>::compressed;

/*=========================== Brick Header Parsing ============================*/
constexpr uint32_t MAGIC = 0x4C5A4331U;  // 'LZC1'
constexpr uint32_t HDR_BYTES = 33280;
constexpr uint32_t MAX_CHUNKS = 2048;
enum DType : uint8_t { DT_DOUBLE = 1, DT_SINGLE = 2, DT_UINT16 = 3 };

struct BrickHeader {
    uint32_t magic;
    uint8_t dtype, ndims;
    uint64_t dims[16];
    uint64_t totalBytes, chunkBytes;
    uint32_t nChunks;
    uint64_t uLen[MAX_CHUNKS], cLen[MAX_CHUNKS];
    uint8_t _pad[HDR_BYTES - (4 + 1 + 1 + 16*8 + 8 + 8 + 4 + MAX_CHUNKS*16)];
};

static void freadExact(FILE* fp, void* dst, std::size_t n, const char* ctx) {
    if (std::fread(dst, 1, n, fp) != n)
        throw std::runtime_error(std::string(ctx) + ": I/O error");
}

static BrickHeader readHeader(FILE* fp, const std::string& file) {
    BrickHeader h{};
    freadExact(fp, &h, HDR_BYTES, "header");
    if (h.magic != MAGIC) throw std::runtime_error(file + ": bad magic");
    if (h.dtype != DT_SINGLE) throw std::runtime_error(file + ": not float32");
    if (h.nChunks == 0 || h.nChunks > MAX_CHUNKS)
        throw std::runtime_error(file + ": bad chunk count");
    return h;
}

inline uint64_t idx3d(uint64_t x, uint64_t y, uint64_t z,
                      uint64_t dimX, uint64_t dimY) {
    return x + dimX * (y + dimY * z);
}

/*============================= Brick Job Struct ==============================*/
template<typename OUT_T>
struct BrickJob {
    std::string file;
    uint64_t x0,y0,z0,x1,y1,z1, dimX,dimY,dimZ;
    OUT_T* dst;
    float scal, ampl, lowClip, highClip, dmin, dmax;
    bool clip;

    void operator()() const {
        auto& bufferFloat = ThreadScratch<float>::uncompressed;
        auto& bufferCompressed = ThreadScratch<char>::compressed;

        std::unique_ptr<FILE, decltype(&std::fclose)> fp(std::fopen(file.c_str(), "rb"), &std::fclose);
        if (!fp) throw std::runtime_error("open " + file);

        const auto header = readHeader(fp.get(), file);

        uint64_t bx = x1 - x0 + 1, by = y1 - y0 + 1, bz = z1 - z0 + 1;
        uint64_t voxelCount = bx * by * bz;
        if (bufferFloat.size() < voxelCount) bufferFloat.resize(voxelCount);

        char* decompressed = reinterpret_cast<char*>(bufferFloat.data());
        uint64_t offset = 0;

        for (uint32_t i = 0; i < header.nChunks; ++i) {
            if (bufferCompressed.size() < header.cLen[i])
                bufferCompressed.resize(header.cLen[i]);
            freadExact(fp.get(), bufferCompressed.data(), header.cLen[i], "chunk");
            int written = LZ4_decompress_safe(bufferCompressed.data(), decompressed + offset,
                                              int(header.cLen[i]), int(header.uLen[i]));
            if (written < 0 || uint64_t(written) != header.uLen[i])
                throw std::runtime_error(file + ": LZ4 error");
            offset += header.uLen[i];
        }

        if (offset != header.totalBytes)
            throw std::runtime_error(file + ": size mismatch");

        const float span = highClip - lowClip;
        const float kClip = clip ? scal * ampl / span : 0.f;
        const float kLinear = scal * ampl / dmax;
        const float kMinMax = scal * ampl / (dmax - dmin);
        const bool useMinMax = (!clip && dmin > 0.f);

        const float* src = bufferFloat.data();
        for (uint64_t z = 0; z < bz; ++z)
        for (uint64_t y = 0; y < by; ++y) {
            const uint64_t base = idx3d(x0, y0 + y, z0 + z, dimX, dimY);
            for (uint64_t x = 0; x < bx; ++x) {
                float val = src[(z * by + y) * bx + x];
                if (clip)
                    val = std::clamp(val - lowClip, 0.f, span) * kClip;
                else if (useMinMax)
                    val = (val - dmin) * kMinMax;
                else
                    val = val * kLinear;

                val -= ampl;
                val = (val >= 0.f) ? std::floor(val + 0.5f) : std::ceil(val - 0.5f);
                val = std::clamp(val, 0.f, scal);
                dst[base + x] = static_cast<OUT_T>(val);
            }
        }
    }
};

/*============================ Atomic Thread Pool =============================*/
template<typename JobT>
void run_atomic_thread_pool(const std::vector<JobT>& jobs, int nThreads) {
    std::atomic<size_t> jobIndex{0};
    std::atomic<bool> hasException{false};
    std::string errorMsg;

    auto worker = [&]() {
        while (true) {
            size_t idx = jobIndex.fetch_add(1, std::memory_order_relaxed);
            if (idx >= jobs.size() || hasException.load()) break;
            try {
                jobs[idx]();
            } catch (const std::exception& e) {
                if (!hasException.exchange(true))
                    errorMsg = e.what();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (hasException)
        throw std::runtime_error(errorMsg);
}

/*============================= MEX Entry Point ================================*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    auto now = std::chrono::high_resolution_clock::now();
    if (nrhs < 12) mexErrMsgTxt("Expected 12 input arguments.");
    if (!mxIsCell(prhs[0])) mexErrMsgTxt("filenames must be a cell array.");

    const mwSize nFiles = mxGetNumberOfElements(prhs[0]);

    auto dimAt = [&](const mxArray* a, int i) -> uint64_t {
        return mxIsUint64(a)
            ? reinterpret_cast<const uint64_t*>(mxGetData(a))[i]
            : uint64_t(mxGetPr(a)[i]);
    };
    uint64_t dimX = dimAt(prhs[3], 0), dimY = dimAt(prhs[3], 1), dimZ = dimAt(prhs[3], 2);

    const bool clip = mxGetScalar(prhs[4]) > 0;
    float scal = float(mxGetScalar(prhs[5]));
    float ampl = float(mxGetScalar(prhs[6]));
    float dmin = float(mxGetScalar(prhs[7]));
    float dmax = float(mxGetScalar(prhs[8]));
    float lowClip = float(mxGetScalar(prhs[9]));
    float highClip = float(mxGetScalar(prhs[10]));
    int maxThreads = (nrhs > 11) ? int(mxGetScalar(prhs[11])) : int(std::thread::hardware_concurrency());
    if (maxThreads < 1) maxThreads = 1;

    const mwSize outDims[3] = { mwSize(dimX), mwSize(dimY), mwSize(dimZ) };
    mxClassID outType = (scal <= 255) ? mxUINT8_CLASS : mxUINT16_CLASS;
    mxArray* out = mxCreateNumericArray(3, outDims, outType, mxREAL);
    void* vol = mxGetData(out);

    const mxArray *p1 = prhs[1], *p2 = prhs[2];
    auto getCoord = [&](const mxArray* a, mwSize idx) -> uint64_t {
        return mxIsUint64(a)
            ? reinterpret_cast<const uint64_t*>(mxGetData(a))[idx]
            : uint64_t(mxGetPr(a)[idx]);
    };

    if (outType == mxUINT8_CLASS) {
        std::vector<BrickJob<uint8_t>> jobs;
        jobs.reserve(nFiles);
        for (mwSize i = 0; i < nFiles; ++i) {
            char* f = mxArrayToUTF8String(mxGetCell(prhs[0], i));
            std::string file(f); mxFree(f);
            jobs.push_back({
                file,
                getCoord(p1,i)-1, getCoord(p1,i+nFiles)-1, getCoord(p1,i+2*nFiles)-1,
                getCoord(p2,i)-1, getCoord(p2,i+nFiles)-1, getCoord(p2,i+2*nFiles)-1,
                dimX, dimY, dimZ, static_cast<uint8_t*>(vol),
                scal, ampl, lowClip, highClip, dmin, dmax, clip });
        }
        try {
            run_atomic_thread_pool(jobs, maxThreads);
        } catch (const std::exception& e) {
            mexErrMsgTxt(e.what());
        }
    } else {
        std::vector<BrickJob<uint16_t>> jobs;
        jobs.reserve(nFiles);
        for (mwSize i = 0; i < nFiles; ++i) {
            char* f = mxArrayToUTF8String(mxGetCell(prhs[0], i));
            std::string file(f); mxFree(f);
            jobs.push_back({
                file,
                getCoord(p1,i)-1, getCoord(p1,i+nFiles)-1, getCoord(p1,i+2*nFiles)-1,
                getCoord(p2,i)-1, getCoord(p2,i+nFiles)-1, getCoord(p2,i+2*nFiles)-1,
                dimX, dimY, dimZ, static_cast<uint16_t*>(vol),
                scal, ampl, lowClip, highClip, dmin, dmax, clip });
        }
        try {
            run_atomic_thread_pool(jobs, maxThreads);
        } catch (const std::exception& e) {
            mexErrMsgTxt(e.what());
        }
    }

    plhs[0] = out;
    if (nlhs > 1) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - now).count();
        plhs[1] = mxCreateDoubleScalar(elapsed);
    }
}
