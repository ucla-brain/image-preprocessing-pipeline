// gauss3d_mex.cu - Robust 3D Gaussian filtering (constant memory, explicit kernel specializations)
// Supports single/double precision MATLAB gpuArray input.
// Uses one workspace buffer, in-place for last axis, and constant memory for the Gaussian kernel.

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>  // <-- Add for half support

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

// MAX_KERNEL_SIZE: Increase to support larger sigmas (up to 64KB constant memory per kernel array)
#define MAX_KERNEL_SIZE 51

__constant__ __half const_kernel_h[MAX_KERNEL_SIZE];
__constant__ float const_kernel_f[MAX_KERNEL_SIZE];
__constant__ double const_kernel_d[MAX_KERNEL_SIZE];

__global__ void gauss1d_kernel_const_half(
    const __half* src, __half* dst,
    int nx, int ny, int nz,
    int klen, int axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    int line = tid / linelen;
    int pos = tid % linelen;

    int x, y, z;
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

    int idx = x + y * nx + z * nx * ny;
    int r = klen / 2;
    float acc = 0.0f;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = x, yi = y, zi = z;
        if (axis == 0) xi = min(max(x + offset, 0), nx - 1);
        if (axis == 1) yi = min(max(y + offset, 0), ny - 1);
        if (axis == 2) zi = min(max(z + offset, 0), nz - 1);
        int src_idx = xi + yi * nx + zi * nx * ny;
        acc += __half2float(src[src_idx]) * __half2float(const_kernel_h[s]);
    }
    dst[idx] = __float2half(acc);
}

// Helper: Cast float device array to half, in-place
__global__ void float_to_half_kernel(const float* src, __half* dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = __float2half(src[idx]);
}

// Helper: Cast half device array to float, in-place
__global__ void half_to_float_kernel(const __half* src, float* dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = __half2float(src[idx]);
}

// Gaussian kernel creation
template <typename T>
void make_gaussian_kernel(T sigma, int ksize, T* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = (T)std::exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i) kernel[i] = (T)(kernel[i] / sum);
}

// Host orchestration for half precision
void gauss3d_separable_half(
    float* input,
    float* buffer,
    int nx, int ny, int nz,
    const float sigma[3], const int ksize[3])
{
    int max_klen = std::max({ksize[0], ksize[1], ksize[2]});
    if (max_klen > MAX_KERNEL_SIZE) {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    }
    size_t N = (size_t)nx * ny * nz;

    // Allocate device __half buffers
    __half* d_a;
    __half* d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__half)));

    // Convert input float -> half
    float_to_half_kernel<<<(N+255)/256,256>>>(input, d_a, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    __half* src = d_a;
    __half* dst = d_b;
    __half* tmp;

    // Kernel in float, cast to half
    float* h_kernel = new float[max_klen];
    __half* h_kernel_h = new __half[max_klen];

    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);
        for (int i = 0; i < ksize[axis]; ++i)
            h_kernel_h[i] = __float2half(h_kernel[i]);
        CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_h, h_kernel_h, ksize[axis]*sizeof(__half), 0, cudaMemcpyHostToDevice));

        int linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        int nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        int total = linelen * nline;
        int block = 256;
        int grid = (total + block - 1) / block;

        gauss1d_kernel_const_half<<<grid, block, 0>>>(src, dst, nx, ny, nz, ksize[axis], axis);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        tmp = src; src = dst; dst = tmp;
    }
    // After three passes, src points to the result
    // Convert back to float
    half_to_float_kernel<<<(N+255)/256,256>>>(src, input, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    delete[] h_kernel;
    delete[] h_kernel_h;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

// CUDA 1D convolution kernels (constant memory only)
__global__ void gauss1d_kernel_const_float(
    const float* src, float* dst,
    int nx, int ny, int nz,
    int klen, int axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    int line = tid / linelen;
    int pos = tid % linelen;

    int x, y, z;
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

    int idx = x + y * nx + z * nx * ny;
    int r = klen / 2;
    float acc = 0.0f;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = x, yi = y, zi = z;
        if (axis == 0) xi = min(max(x + offset, 0), nx - 1);
        if (axis == 1) yi = min(max(y + offset, 0), ny - 1);
        if (axis == 2) zi = min(max(z + offset, 0), nz - 1);
        int src_idx = xi + yi * nx + zi * nx * ny;
        acc += src[src_idx] * const_kernel_f[s];
    }
    dst[idx] = acc;
}

__global__ void gauss1d_kernel_const_double(
    const double* src, double* dst,
    int nx, int ny, int nz,
    int klen, int axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    int line = tid / linelen;
    int pos = tid % linelen;

    int x, y, z;
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

    int idx = x + y * nx + z * nx * ny;
    int r = klen / 2;
    double acc = 0.0;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = x, yi = y, zi = z;
        if (axis == 0) xi = min(max(x + offset, 0), nx - 1);
        if (axis == 1) yi = min(max(y + offset, 0), ny - 1);
        if (axis == 2) zi = min(max(z + offset, 0), nz - 1);
        int src_idx = xi + yi * nx + zi * nx * ny;
        acc += src[src_idx] * const_kernel_d[s];
    }
    dst[idx] = acc;
}
template <>
void gauss3d_separable<__half>(
    __half* input,
    __half* buffer,
    int nx, int ny, int nz,
    const __half sigma[3], const int ksize[3])
{
    int max_klen = std::max({ksize[0], ksize[1], ksize[2]});
    if (max_klen > MAX_KERNEL_SIZE) {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    }
    float* h_kernel_f = new float[max_klen];
    __half* h_kernel_h = new __half[max_klen];

    __half* src = input;
    __half* dst = buffer;
    float sigma_f[3] = { __half2float(sigma[0]), __half2float(sigma[1]), __half2float(sigma[2]) };

    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma_f[axis], ksize[axis], h_kernel_f);
        for (int i = 0; i < ksize[axis]; ++i)
            h_kernel_h[i] = __float2half(h_kernel_f[i]);
        CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_h, h_kernel_h, ksize[axis] * sizeof(__half), 0, cudaMemcpyHostToDevice));

        int linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        int nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        int total = linelen * nline;
        int block = 256;
        int grid = (total + block - 1) / block;

        gauss1d_kernel_const_half<<<grid, block, 0>>>(src, dst, nx, ny, nz, ksize[axis], axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(src, dst);
    }
    if (src != input) {
        CUDA_CHECK(cudaMemcpy(input, src, (size_t)nx * ny * nz * sizeof(__half), cudaMemcpyDeviceToDevice));
    }
    delete[] h_kernel_f;
    delete[] h_kernel_h;
}

// Host orchestration with 1 buffer, in-place for last axis
template <typename T>
void gauss3d_separable(
    T* input,
    T* buffer,
    int nx, int ny, int nz,
    const T sigma[3], const int ksize[3])
{
    int max_klen = std::max({ksize[0], ksize[1], ksize[2]});
    if (max_klen > MAX_KERNEL_SIZE) {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    }
    T* h_kernel = new T[max_klen];

    T* src = input;
    T* dst = buffer;

    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);

        int linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        int nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        int total = linelen * nline;
        int block = 256;
        int grid = (total + block - 1) / block;

        if (std::is_same<T, float>::value) {
            CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_f, h_kernel, ksize[axis] * sizeof(float), 0, cudaMemcpyHostToDevice));
            gauss1d_kernel_const_float<<<grid, block, 0>>>(
                reinterpret_cast<const float*>(src),
                reinterpret_cast<float*>(dst),
                nx, ny, nz, ksize[axis], axis);
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_d, h_kernel, ksize[axis] * sizeof(double), 0, cudaMemcpyHostToDevice));
            gauss1d_kernel_const_double<<<grid, block, 0>>>(
                reinterpret_cast<const double*>(src),
                reinterpret_cast<double*>(dst),
                nx, ny, nz, ksize[axis], axis);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(src, dst);
    }
    // After 3 axes, src points to the result (due to odd number of swaps)
    if (src != input) {
        CUDA_CHECK(cudaMemcpy(input, src, (size_t)nx * ny * nz * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    delete[] h_kernel;
}

// ================
// MEX entry point
// ================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2)
        mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size, 'half'])");

    const mxGPUArray* img_gpu_const = mxGPUCreateFromMxArray(prhs[0]);
    void* ptr = mxGPUGetData(const_cast<mxGPUArray*>(img_gpu_const));
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3)
        mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D.");

    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];
    size_t N = (size_t)nx * ny * nz;
    mxClassID cls = mxGPUGetClassID(img_gpu);
    void* ptr = mxGPUGetData(img_gpu);
    void* buffer = nullptr;

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
        if (mxIsScalar(prhs[2])) {
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

    // --- Parse optional 'half' mode ---
    bool use_half = false;
    if (nrhs >= 4 && mxIsChar(prhs[3])) {
        char mode[16];
        mxGetString(prhs[3], mode, sizeof(mode));
        if (strcmp(mode, "half") == 0) use_half = true;
    }

    // --- Dispatch by type ---
    if (use_half) {
        if (cls != mxSINGLE_CLASS) {
            mexErrMsgIdAndTxt("gauss3d:half", "'half' mode requires single-precision gpuArray input.");
        }
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(float)));
        float sigma[3] = { (float)sigma_double[0], (float)sigma_double[1], (float)sigma_double[2] };
        gauss3d_separable_half((float*)ptr, (float*)buffer, nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaFree(buffer));
    } else if (cls == mxSINGLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(float)));
        float sigma[3] = { (float)sigma_double[0], (float)sigma_double[1], (float)sigma_double[2] };
        gauss3d_separable<float>((float*)ptr, (float*)buffer, nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaFree(buffer));
    } else if (cls == mxDOUBLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(double)));
        double sigma[3] = { sigma_double[0], sigma_double[1], sigma_double[2] };
        gauss3d_separable<double>((double*)ptr, (double*)buffer, nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaFree(buffer));
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
