/*
 * deconFFT_gpu.cu
 * Author: Keivan Moradi, ChatGPT 4.1 (2025)
 * License: GPL v3
 *
 * GPU-accelerated deconvolution with regularization for 3D single-precision gpuArray inputs.
 * Computes:
 *   buf = convFFT(bl, otf);
 *   buf = max(buf, eps('single'));
 *   buf = bl ./ buf;
 *   buf = convFFT(buf, otf_conj);
 *   if (lambda > 0)
 *     reg = convFFT(bl, R_fft); % R_fft = fftn(R, size(bl)), if R provided
 *     bl = bl .* buf .* (1 - lambda) + reg .* lambda;
 *   else
 *     bl = bl .* buf;
 *   end
 *   bl = abs(bl);
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdint>
#include <limits>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("deconFFT_mex:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) \
        mexErrMsgIdAndTxt("deconFFT_mex:cufft", "CUFFT error %s:%d: %d", __FILE__, __LINE__, (int)err); \
} while(0)

#define EPS_SINGLE 1.1920929e-07f

__global__ void max_with_eps(float* a, size_t n, float eps) {
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) a[idx] = fmaxf(a[idx], eps);
}
__global__ void safe_divide(float* a, const float* b, size_t n, float eps) {
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float denom = fmaxf(b[idx], eps);
        a[idx] = a[idx] / denom;
    }
}
__global__ void abs_in_place(float* a, size_t n) {
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) a[idx] = fabsf(a[idx]);
}
__global__ void product(float* bl, const float* buf, size_t n) {
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) bl[idx] = bl[idx] * buf[idx];
}
__global__ void weighted_sum(float* bl, const float* buf, const float* reg, float lambda, size_t n) {
    float one_minus_lambda = 1.0f - lambda;
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) bl[idx] = bl[idx] * buf[idx] * one_minus_lambda + reg[idx] * lambda;
}
__global__ void multiply_complex(cufftComplex* a, const cufftComplex* b, size_t n) {
    size_t idx = size_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < n) {
        cufftComplex av = a[idx], bv = b[idx];
        cufftComplex out;
        out.x = av.x * bv.x - av.y * bv.y;
        out.y = av.x * bv.y + av.y * bv.x;
        a[idx] = out;
    }
}

void convFFT(float* d_in, const cufftComplex* d_otf, float* d_out,
             int dx, int dy, int dz, cufftHandle plan_fwd, cufftHandle plan_inv) {
    size_t nvox = size_t(dx) * dy * dz;
    cufftComplex* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(cufftComplex) * nvox));
    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_in, d_buf));
    size_t nthreads = 256;
    size_t nblocks = (nvox + nthreads - 1) / nthreads;
    multiply_complex<<<nblocks, nthreads>>>(d_buf, d_otf, nvox);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_buf, d_out));
    CUDA_CHECK(cudaFree(d_buf));
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 4)
        mexErrMsgIdAndTxt("deconFFT_mex:nrhs", "Requires at least 4 inputs: bl, otf, otf_conj, lambda");

    // All inputs as mxGPUArray* (no const)
    mxGPUArray* bl_gpu         = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray* otf_gpu        = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray* otf_conj_gpu   = mxGPUCreateFromMxArray(prhs[2]);
    float lambda = *(float*)mxGetData(prhs[3]);

    const mwSize* sz = mxGPUGetDimensions(bl_gpu);
    int dx = (int)sz[0], dy = (int)sz[1], dz = (int)sz[2];
    size_t nvox = size_t(dx) * dy * dz;

    if (nvox > std::numeric_limits<size_t>::max())
        mexErrMsgIdAndTxt("deconFFT_mex:size", "Array size too large for size_t.");

    float* d_bl = (float*)mxGPUGetData(bl_gpu); // IN-PLACE!
    cufftComplex* d_otf = (cufftComplex*)mxGPUGetDataReadOnly(otf_gpu); // treat as read-only
    cufftComplex* d_otf_conj = (cufftComplex*)mxGPUGetDataReadOnly(otf_conj_gpu); // treat as read-only

    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan3d(&plan_fwd, dz, dy, dx, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&plan_inv, dz, dy, dx, CUFFT_C2R));

    float* d_buf; CUDA_CHECK(cudaMalloc(&d_buf, nvox * sizeof(float)));
    float* d_reg = nullptr;
    if (lambda > 0) CUDA_CHECK(cudaMalloc(&d_reg, nvox * sizeof(float)));

    size_t nthreads = 256;
    size_t nblocks = (nvox + nthreads - 1) / nthreads;

    // buf = convFFT(bl, otf)
    convFFT(d_bl, d_otf, d_buf, dx, dy, dz, plan_fwd, plan_inv);

    // buf = max(buf, eps)
    max_with_eps<<<nblocks, nthreads>>>(d_buf, nvox, EPS_SINGLE);

    // buf = bl ./ buf (store result in bl)
    safe_divide<<<nblocks, nthreads>>>(d_bl, d_buf, nvox, EPS_SINGLE);

    // buf = convFFT(bl, otf_conj) (store result in d_buf)
    convFFT(d_bl, d_otf_conj, d_buf, dx, dy, dz, plan_fwd, plan_inv);

    if (lambda > 0) {
        CUDA_CHECK(cudaMemcpy(d_reg, d_bl, nvox * sizeof(float), cudaMemcpyDeviceToDevice));
        weighted_sum<<<nblocks, nthreads>>>(d_bl, d_buf, d_reg, lambda, nvox);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_reg));
    } else {
        product<<<nblocks, nthreads>>>(d_bl, d_buf, nvox);
        CUDA_CHECK(cudaGetLastError());
    }

    abs_in_place<<<nblocks, nthreads>>>(d_bl, nvox);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_buf));
    CUFFT_CHECK(cufftDestroy(plan_fwd));
    CUFFT_CHECK(cufftDestroy(plan_inv));

    plhs[0] = mxGPUCreateMxArrayOnGPU(bl_gpu);
    mxGPUDestroyGPUArray(bl_gpu);
    mxGPUDestroyGPUArray(otf_gpu);
    mxGPUDestroyGPUArray(otf_conj_gpu);
}
