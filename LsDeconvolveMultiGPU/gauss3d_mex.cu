// gauss3d_mex.cu - Single-precision 3D Gaussian filtering for MATLAB gpuArray

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>   // For std::this_thread::sleep_for
#include <chrono>   // For std::chrono::milliseconds

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexWarnMsgIdAndTxt("gauss3d:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
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
        mexWarnMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
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
            mexWarnMsgIdAndTxt("gauss3d:cuda", "CUDA memcpyToSymbol error: %s", cudaGetErrorString(err));
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
            mexWarnMsgIdAndTxt("gauss3d:cuda", "CUDA kernel launch error: %s", cudaGetErrorString(err));
            local_error = true;
            break;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d:cuda", "CUDA device synchronize error: %s", cudaGetErrorString(err));
            local_error = true;
            break;
        }
        std::swap(src, dst);
    }

    if (!local_error && src != input) {
        cudaError_t err = cudaMemcpy(input, src, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            mexWarnMsgIdAndTxt("gauss3d:cuda", "CUDA memcpy result error: %s", cudaGetErrorString(err));
            local_error = true;
        }
    }

    delete[] h_kernel;
    if (error_flag) *error_flag = local_error;
}

// ================
// MEX entry point
// ================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();
    float* buffer = nullptr;
    bool error_flag = false;
    mxGPUArray* img_gpu = nullptr;
    mxGPUArray* out_gpu = nullptr;

    try {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size])");

        img_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[0]);
        const mwSize* sz = mxGPUGetDimensions(img_gpu);
        int nd = mxGPUGetNumberOfDimensions(img_gpu);
        if (nd != 3)
            mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D.");

        size_t nx = (size_t)sz[0], ny = (size_t)sz[1], nz = (size_t)sz[2];
        size_t N = nx * ny * nz;
        mxClassID cls = mxGPUGetClassID(img_gpu);
        void* ptr = mxGPUGetData(img_gpu);

        if (cls != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("gauss3d:class", "Input must be single-precision gpuArray");

        double sigma_double[3];
        if (mxIsScalar(prhs[1])) {
            double v = mxGetScalar(prhs[1]);
            sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
        } else if (mxGetNumberOfElements(prhs[1]) == 3) {
            double* s = mxGetPr(prhs[1]);
            for (int i = 0; i < 3; ++i) sigma_double[i] = s[i];
        } else {
            mexErrMsgIdAndTxt("gauss3d:sigma", "sigma must be scalar or 3-vector");
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
                mexErrMsgIdAndTxt("gauss3d:kernel", "kernel_size must be scalar or 3-vector");
            }
        } else {
            for (int i = 0; i < 3; ++i)
                ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
        }

        // --------- OOM-aware cudaMalloc with retry -----------
        int max_retries = 2;
        int retries = 0;
        cudaError_t alloc_err;
        while (retries < max_retries) {
            alloc_err = cudaMalloc(&buffer, N * sizeof(float));
            if (alloc_err == cudaSuccess && buffer != nullptr) {
                break;
            } else {
                size_t free_bytes = 0, total_bytes = 0;
                cudaMemGetInfo(&free_bytes, &total_bytes);
                mexWarnMsgIdAndTxt("gauss3d:cuda",
                    "CUDA OOM: Tried to allocate %.2f MB (Free: %.2f MB). Attempt %d/%d.",
                    N * sizeof(float) / 1024.0 / 1024.0,
                    free_bytes / 1024.0 / 1024.0,
                    retries+1, max_retries);
                cudaDeviceSynchronize();
                cudaFree(0);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                retries++;
                buffer = nullptr;
            }
        }
        if (alloc_err != cudaSuccess || !buffer) {
            mexWarnMsgIdAndTxt("gauss3d:cuda",
                "CUDA OOM: Could not allocate workspace buffer (%.2f MB) after %d attempts. Try reducing input size or call reset(gpuDevice).",
                N * sizeof(float) / 1024.0 / 1024.0, max_retries);
            error_flag = true;
            goto cleanup;
        }

        float sigma[3] = { (float)sigma_double[0], (float)sigma_double[1], (float)sigma_double[2] };
        gauss3d_separable_float((float*)ptr, buffer, nx, ny, nz, sigma, ksize, &error_flag);

        // Synchronize before returning to MATLAB to catch any lingering errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // Output result (returns the same array, modifies in-place)
        out_gpu = img_gpu;
        plhs[0] = mxGPUCreateMxArrayOnGPU(out_gpu);

    } catch (...) {
        mexPrintf("Unknown error in gauss3d_mex.cu. Possible OOM or kernel failure.\n");
        error_flag = true;
        goto cleanup;
    }

cleanup:
    if (buffer) {
        cudaFree(buffer);
        cudaDeviceSynchronize();  // Make sure all memory is actually freed
        cudaFree(0);
        buffer = nullptr;
    }
    // Do not destroy img_gpu if returned to MATLAB
    if (error_flag) {
        // Additional error logging can go here
    }
}
