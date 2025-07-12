/*
    gauss3d_gpu.cu

    High-performance, in-place 3D Gaussian filtering for MATLAB gpuArray inputs
    using CUDA. Supports both separable direct and FFT-based filtering.

    ----------------------------------------------------------------------------
    Author:       Keivan Moradi (with assistance from ChatGPT v4.1, 2025)
    License:      GPL v3
    ----------------------------------------------------------------------------

    Overview:
    ---------
    This MEX function implements a fast, VRAM-efficient **3D Gaussian filter**
    for single-precision (`single`) 3D `gpuArray` data in MATLAB, with
    CUDA acceleration. It is compatible with MATLAB's `imgaussfilt3`, but
    optimized for large volumes, pipelined workflows, and multi-step GPU pipelines.

    **IMPORTANT:** Filtering is performed IN-PLACE. The input gpuArray is
    overwritten and returned as output. The original input data is destroyed.

    Features:
    ---------
      - **In-place destructive filtering:** Input array is modified directly;
        only a single workspace buffer is allocated in addition.
      - **Two filtering methods:**
          • **Direct (separable):** Three-pass 1D Gaussian convolution using
            CUDA constant memory, optimal for small–moderate kernel sizes.
          • **FFT-based:** Full 3D convolution via cuFFT for large kernels/volumes.
        The method is auto-selected based on size, or can be set by user.
      - **Automatic out-of-memory handling:** Allocations are retried with
        informative warnings.
      - **Flexible parameters:** Accepts scalar or 3-vector `sigma`, with
        automatic or explicit kernel size selection.
      - **MATLAB integration:** Both input and output are MATLAB `gpuArray(single)`.
      - **Fully open source, GPL v3.**

    Differences from MATLAB's imgaussfilt3:
    ---------------------------------------
      1. **In-place/destructive:** The input gpuArray is overwritten.
      2. **Explicit VRAM management:** Workspace buffer is managed internally
         with OOM resilience and optional user control.
      3. **Auto or explicit method selection:** FFT is chosen for large
         kernels/volumes; user can override.
      4. **Optimized for pipelined/block-wise use:** Designed to slot into
         large-scale, multi-GPU deconvolution pipelines.
      5. **No CPU fallback:** All operations require GPU memory and CUDA.

    Usage:
    ------
        % Standard usage
        x = gpuArray(single(randn(128,128,64)));
        y = gauss3d_gpu(x, 2.0);                        % Isotropic, auto kernel size/mode
        y = gauss3d_gpu(x, [2 1 4], [9 5 15]);          % Anisotropic, user kernel size

        % Forcing FFT or direct mode (optional 4th arg: 'fft' or 'direct')
        y = gauss3d_gpu(x, 2.0, [], 'fft');             % Force FFT mode
        y = gauss3d_gpu(x, 2.0, [], 'direct');          % Force direct convolution

    Arguments:
    ----------
      x           : 3D `gpuArray(single)`, input (overwritten)
      sigma       : Scalar or 3-vector, Gaussian sigma(s) [default: required]
      kernel_size : Scalar or 3-vector, optional kernel size (odd, per axis)
                    Default: 2*ceil(3*sigma)+1 per axis
      mode        : (Optional, char) 'fft', 'direct', or omitted for auto

    Notes:
    ------
      - **Input is 3D, single-precision, gpuArray.**
      - **Destructive:** The input is destroyed; output is the filtered array.
      - Workspace buffer of same size as input is allocated temporarily.
      - The function auto-switches between FFT and direct mode, but can be forced.
      - Optimized for high-throughput, block-wise, or batch GPU workflows.
      - Not intended for small images or use on CPU data.

    Acknowledgments:
    ----------------
      - Algorithm and optimization: Keivan Moradi
      - CUDA/MEX structure review: ChatGPT (OpenAI GPT-4.1, 2025)

*/


#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>   // For std::this_thread::sleep_for
#include <chrono>   // For std::chrono::milliseconds
#include <cufft.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexWarnMsgIdAndTxt("gauss3d_gpu:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

#define MAX_KERNEL_SIZE 51
__constant__ float const_kernel_f[MAX_KERNEL_SIZE];

// Gaussian kernel creation (host)
void make_gaussian_kernel(float sigma, int ksize, float* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] = static_cast<float>(kernel[i] / sum);
}

// CUDA 1D convolution kernel for float (with restrict)
__global__ void gauss1d_kernel_const_float(
    const float* __restrict__ src, float* __restrict__ dst,
    size_t nx, size_t ny, size_t nz,
    int klen, int axis)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    size_t line = tid / linelen;
    size_t pos = tid % linelen;

    size_t x, y, z;
    if (axis == 0) {
        y = line % ny;
        z = line / ny;
        x = pos;
    } else if (axis == 1) {
        x = line % nx;
        z = line / nx;
        y = pos;
    } else {
        x = line % nx;
        y = line / nx;
        z = pos;
    }

    size_t idx = x + y * nx + z * nx * ny;
    int r = klen / 2;
    float acc = 0.0f;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);
        int zi = static_cast<int>(z);
        if (axis == 0) xi = min(max(static_cast<int>(x) + offset, 0), static_cast<int>(nx) - 1);
        if (axis == 1) yi = min(max(static_cast<int>(y) + offset, 0), static_cast<int>(ny) - 1);
        if (axis == 2) zi = min(max(static_cast<int>(z) + offset, 0), static_cast<int>(nz) - 1);
        size_t src_idx = xi + yi * nx + zi * nx * ny;
        acc += src[src_idx] * const_kernel_f[s];
    }
    dst[idx] = acc;
}

// Host orchestration for float
void gauss3d_separable_float(
    float* input,
    float* buffer,
    size_t nx, size_t ny, size_t nz,
    const float sigma[3], const int ksize[3],
    bool* error_flag)
{
    int max_klen = std::max({ksize[0], ksize[1], ksize[2]});
    if (max_klen > MAX_KERNEL_SIZE) {
        mexWarnMsgIdAndTxt("gauss3d_gpu:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
        if (error_flag) *error_flag = true;
        return;
    }
    float* h_kernel = new float[max_klen];
    float* src = input;
    float* dst = buffer;
    bool local_error = false;

    // --- Use cudaOccupancyMaxPotentialBlockSize for kernel launch tuning ---
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, gauss1d_kernel_const_float, 0, 0);

    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);
        cudaError_t err = cudaMemcpyToSymbol(const_kernel_f, h_kernel, ksize[axis] * sizeof(float), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d_gpu:cuda", "CUDA memcpyToSymbol error: %s", cudaGetErrorString(err));
            local_error = true;
            break;
        }
        size_t linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        size_t nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        size_t total = linelen * nline;
        int grid = static_cast<int>((total + blockSize - 1) / blockSize);

        gauss1d_kernel_const_float<<<grid, blockSize, 0>>>(
            src, dst, nx, ny, nz, ksize[axis], axis);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d_gpu:cuda", "CUDA kernel launch error: %s", cudaGetErrorString(err));
            local_error = true;
            break;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d_gpu:cuda", "CUDA device synchronize error: %s", cudaGetErrorString(err));
            local_error = true;
            break;
        }
        std::swap(src, dst);
    }

    if (!local_error && src != input) {
        cudaError_t err = cudaMemcpy(input, src, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d_gpu:cuda", "CUDA memcpy result error: %s", cudaGetErrorString(err));
            local_error = true;
        }
    }

    delete[] h_kernel;
    if (error_flag) *error_flag = local_error;
}

// A small helper to check CUFFT calls:
#define CUFFT_CHECK(call) do {                            \
    cufftResult err = call;                               \
    if (err != CUFFT_SUCCESS) {                           \
        mexWarnMsgIdAndTxt(                               \
            "gauss3d_gpu:fft",                            \
            "CUFFT error %d at %s:%d",                    \
            (int)err, __FILE__, __LINE__                  \
        );                                                \
        goto cleanup;                                     \
    }                                                     \
} while(0)

// --- CUDA kernel helpers ---
__global__ void mult_freq_domain(cufftComplex* a, const cufftComplex* b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        cufftComplex x = a[i], y = b[i];
        a[i].x = x.x*y.x - x.y*y.y;
        a[i].y = x.x*y.y + x.y*y.x;
    }
}
__global__ void scale_array(float* a, int n, float scale) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) a[i] *= scale;
}

// Column-major linear index: row + col*rows + slice*rows*cols
static inline __host__ __device__
size_t cmIdx(int row,int col,int z,int rows,int cols)
{
    return static_cast<size_t>(row)
         + static_cast<size_t>(col)*rows
         + static_cast<size_t>(z)  *rows*cols;
}

// -----------------------------------------------------------------------------
// Build a full-volume Gaussian kernel:
//
// • Truncate at user-supplied ksize (half-widths rx,ry,rz)
// • Normalise so sum == 1
// • Circularly shift by −⌊N/2⌋ in each dimension so the centre lands at (0,0,0)
// • Memory layout matches MATLAB (column-major).
// -----------------------------------------------------------------------------
void make_gaussian_kernel_fft(float*       kernel,
                              int          nRows,    // MATLAB dim-1  (Y)
                              int          nCols,    // MATLAB dim-2  (X)
                              int          nZ,
                              const float  sigma[3], // σx,σy,σz per imgaussfilt3
                              const int    ksize[3]) // user-requested odd sizes
{
    const int rCentre = nRows / 2;
    const int cCentre = nCols / 2;
    const int zCentre = nZ    / 2;

    const int ry = ksize[1] / 2;   // NOTE: σ[1] is along rows (Y)
    const int rx = ksize[0] / 2;   // σ[0] along columns (X)
    const int rz = ksize[2] / 2;

    double sum = 0.0;

    // -------- fill (row,col,z) ------------------------------------------------
    for (int z = 0; z < nZ; ++z)
    {
        const int dz = z - zCentre;
        const float dz_n = (sigma[2] > 0.f) ? dz / sigma[2] : 0.f;

        for (int col = 0; col < nCols; ++col)
        {
            const int dc = col - cCentre;
            const float dc_n = (sigma[0] > 0.f) ? dc / sigma[0] : 0.f;

            for (int row = 0; row < nRows; ++row)
            {
                const int dr = row - rCentre;
                float val = 0.f;

                if (std::abs(dc) <= rx &&
                    std::abs(dr) <= ry &&
                    std::abs(dz) <= rz)
                {
                    const float dr_n = (sigma[1] > 0.f) ? dr / sigma[1] : 0.f;
                    val = expf(-0.5f * (dc_n*dc_n + dr_n*dr_n + dz_n*dz_n));
                }

                kernel[ cmIdx(row,col,z,nRows,nCols) ] = val;
                sum += val;
            }
        }
    }

    // -------- normalise -------------------------------------------------------
    if (sum > 0.0)
    {
        const float invSum = static_cast<float>(1.0 / sum);
        const size_t total = static_cast<size_t>(nRows)*nCols*nZ;
        for (size_t i = 0; i < total; ++i) kernel[i] *= invSum;
    }

    // -------- circular shift by −⌊N/2⌋ --------------------------------------
    const int sRow = nRows / 2;   // MATLAB: floor(N/2)
    const int sCol = nCols / 2;
    const int sZ   = nZ    / 2;

    std::vector<float> tmp(kernel, kernel + static_cast<size_t>(nRows)*nCols*nZ);

    for (int z = 0; z < nZ; ++z)
    {
        const int srcZ = (z + sZ) % nZ;
        for (int col = 0; col < nCols; ++col)
        {
            const int srcC = (col + sCol) % nCols;
            for (int row = 0; row < nRows; ++row)
            {
                const int srcR = (row + sRow) % nRows;
                kernel[ cmIdx(row,col,z,nRows,nCols) ] =
                    tmp   [ cmIdx(srcR,srcC,srcZ,nRows,nCols) ];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// FFT-based Gaussian filtering (periodic boundary conditions).
// Out-of-place, safe for use with MATLAB gpuArray buffers.
// -----------------------------------------------------------------------------
void gauss3d_fft_float(float* d_input, int nx, int ny, int nz,
                       const float sigma[3], const int ksize[3],
                       bool* error_flag)
{
    // ---- Shapes (column-major order): [nx, ny, nz] in MATLAB ===
    // cuFFT expects [nz, ny, nx] (depth, height, width)
    const size_t N      = (size_t)nx * ny * nz;              // Real volume
    const int    NXfreq = nx / 2 + 1;               // freq length along x
    const size_t Nfreq  = (size_t)NXfreq * ny * nz; // ny·nz·(nx/2+1)

    // Host-side printing
    mexPrintf("[gauss3d_gpu] FFT input dims: nx=%d, ny=%d, nz=%d\n", nx, ny, nz);
    mexPrintf("  N=%llu  NXfreq=%d  Nfreq=%llu\n", (unsigned long long)N, NXfreq, (unsigned long long)Nfreq);

    if (error_flag) *error_flag = true;

    float*        d_kernel    = nullptr;
    cufftComplex* d_freq_in   = nullptr;
    cufftComplex* d_freq_ker  = nullptr;
    cufftHandle   planR2C     = 0;
    cufftHandle   planC2R     = 0;
    float*        d_result    = nullptr;

    try {
        // ---- Device allocations ----
        CUDA_CHECK(cudaMalloc(&d_kernel,  sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_freq_in, sizeof(cufftComplex) * Nfreq));
        CUDA_CHECK(cudaMalloc(&d_freq_ker,sizeof(cufftComplex) * Nfreq));
        CUDA_CHECK(cudaMalloc(&d_result,  sizeof(float) * N));

        // ---- Build & upload kernel ----
        std::vector<float> h_kernel(N, 0.f);
        make_gaussian_kernel_fft(h_kernel.data(), nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(),
                              sizeof(float) * N, cudaMemcpyHostToDevice));

        // ---- FFT plans: dims = [nz, ny, nx] ----
        CUFFT_CHECK(cufftPlan3d(&planR2C, nz, ny, nx, CUFFT_R2C));
        CUFFT_CHECK(cufftPlan3d(&planC2R, nz, ny, nx, CUFFT_C2R));

        // ---- Forward FFTs ----
        CUFFT_CHECK(cufftExecR2C(planR2C, d_input,  d_freq_in));
        CUFFT_CHECK(cufftExecR2C(planR2C, d_kernel, d_freq_ker));

        // ---- Point-wise multiply in frequency domain ----
        constexpr int THREADS = 256;
        int blocks = static_cast<int>((Nfreq + THREADS - 1) / THREADS);
        mult_freq_domain<<<blocks, THREADS>>>(d_freq_in, d_freq_ker, static_cast<int>(Nfreq));
        CUDA_CHECK(cudaGetLastError());

        // ---- Inverse FFT and normalize ----
        CUFFT_CHECK(cufftExecC2R(planC2R, d_freq_in, d_result));
        blocks = static_cast<int>((N + THREADS - 1) / THREADS);
        scale_array<<<blocks, THREADS>>>(d_result, static_cast<int>(N), 1.f / float(N));
        CUDA_CHECK(cudaGetLastError());

        // ---- Copy result back to input buffer (for in-place output) ----
        CUDA_CHECK(cudaMemcpy(d_input, d_result, sizeof(float) * N, cudaMemcpyDeviceToDevice));

        if (error_flag) *error_flag = false;

    } catch (...) {
        mexWarnMsgIdAndTxt("gauss3d_gpu:fft", "Exception during FFT filtering");
        if (error_flag) *error_flag = true;
    }

    // ---- Cleanup ----
cleanup:
    if (planR2C)   cufftDestroy(planR2C);
    if (planC2R)   cufftDestroy(planC2R);
    if (d_kernel)  cudaFree(d_kernel);
    if (d_freq_in) cudaFree(d_freq_in);
    if (d_freq_ker)cudaFree(d_freq_ker);
    if (d_result)  cudaFree(d_result);
}


// ================
// MEX entry point
// ================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();
    float* buffer = nullptr;
    bool error_flag = false;
    mxGPUArray* img_gpu = nullptr;

    try {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("gauss3d_gpu:", "Usage: gauss3d_gpu(x, sigma [, kernel_size])");

        img_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[0]);
        const mwSize* sz = mxGPUGetDimensions(img_gpu);
        int nd = mxGPUGetNumberOfDimensions(img_gpu);
        if (nd != 3)
            mexErrMsgIdAndTxt("gauss3d_gpu:", "Input must be 3D.");

        size_t nx = (size_t)sz[0], ny = (size_t)sz[1], nz = (size_t)sz[2];
        size_t N = nx * ny * nz;
        mxClassID cls = mxGPUGetClassID(img_gpu);
        void* ptr = mxGPUGetData(img_gpu);

        if (cls != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("gauss3d_gpu:", "Input must be single-precision gpuArray");

        double sigma_double[3];
        if (mxIsScalar(prhs[1])) {
            double v = mxGetScalar(prhs[1]);
            sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
        } else if (mxGetNumberOfElements(prhs[1]) == 3) {
            double* s = mxGetPr(prhs[1]);
            for (int i = 0; i < 3; ++i) sigma_double[i] = s[i];
        } else {
            mexErrMsgIdAndTxt("gauss3d_gpu:", "sigma must be scalar or 3-vector");
        }

        int ksize[3];
        if (nrhs >= 3 && !mxIsLogicalScalar(prhs[2])) {
            if (mxIsEmpty(prhs[2])) {
                for (int i = 0; i < 3; ++i)
                    ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
            } else if (mxIsScalar(prhs[2])) {
                int k = (int)mxGetScalar(prhs[2]);
                ksize[0] = ksize[1] = ksize[2] = k;
            } else if (mxGetNumberOfElements(prhs[2]) == 3) {
                double* ks = mxGetPr(prhs[2]);
                for (int i = 0; i < 3; ++i) ksize[i] = (int)ks[i];
            } else {
                mexErrMsgIdAndTxt("gauss3d_gpu:", "kernel_size must be scalar or 3-vector");
            }
        } else {
            for (int i = 0; i < 3; ++i)
                ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
        }

        // --------- Allocate workspace buffer only ---------
        int max_retries = 2;
        int retries = 0;
        cudaError_t alloc_err;
        while (retries < max_retries) {
            alloc_err = cudaMalloc(&buffer, N * sizeof(float));
            if (alloc_err == cudaSuccess && buffer != nullptr)
                break;
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            mexWarnMsgIdAndTxt("gauss3d_gpu:cuda",
                "gauss3d_gpu: CUDA OOM: Tried to allocate %.2f MB (Free: %.2f MB). Attempt %d/%d.",
                N * sizeof(float) / 1024.0 / 1024.0,
                free_bytes / 1024.0 / 1024.0,
                retries + 1, max_retries);
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            retries++;
        }
        if (alloc_err != cudaSuccess || !buffer) {
            mexErrMsgIdAndTxt("gauss3d_gpu:cuda",
                "gauss3d_gpu: CUDA OOM: Could not allocate workspace buffer (%.2f MB) after %d attempts.",
                N * sizeof(float) / 1024.0 / 1024.0, max_retries);
        }

        float sigma[3] = { (float)sigma_double[0], (float)sigma_double[1], (float)sigma_double[2] };

        // --------- In-place filtering or FFT! ---------
        float* data_ptr = static_cast<float*>(ptr);
        
        bool use_fft = false;
        int fft_size_thresh = 256; // cube root of 16 million; tune for your GPU
        
        if (nrhs >= 4 && mxIsChar(prhs[3])) {
            char mode_buf[16];
            mxGetString(prhs[3], mode_buf, sizeof(mode_buf));
            if (strcmp(mode_buf, "fft") == 0) use_fft = true;
            else if (strcmp(mode_buf, "direct") == 0) use_fft = false;
            // else auto
        } else {
            // Auto-switch
            use_fft = (nx >= fft_size_thresh || ny >= fft_size_thresh || nz >= fft_size_thresh
                        || std::max({ksize[0], ksize[1], ksize[2]}) > 31);
        }
        
        if (use_fft) {
            gauss3d_fft_float(      data_ptr, nx, ny, nz, sigma, ksize, &error_flag);
        } else {
            gauss3d_separable_float(data_ptr, buffer, nx, ny, nz, sigma, ksize, &error_flag);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Return the (modified) input gpuArray as output
        plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);

    } catch (...) {
        mexPrintf("gauss3d_gpu: Unknown error! Possible OOM or kernel failure.\n");
        error_flag = true;
    }

    // ----------- CLEANUP (always reached) --------------
    if (buffer)
        cudaFree(buffer);
    // Only destroy img_gpu if NOT returned to MATLAB
    // This is robust: if plhs[0] has been set, MATLAB owns it.
    if (img_gpu && (nlhs == 0 || (plhs[0] != mxGPUCreateMxArrayOnGPU(img_gpu))))
        mxGPUDestroyGPUArray(img_gpu);
}
