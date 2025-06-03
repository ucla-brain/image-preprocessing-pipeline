#ifndef CONVOLVE2D_DOWNSAMPLE_CUH
#define CONVOLVE2D_DOWNSAMPLE_CUH

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "mex.h"

// ===============
// Debug Toggle
// ===============
#define DEBUG_WAVELET_CONV

#define TILE 16

// ===============
// Debug Check Macro
// ===============
#ifdef DEBUG_WAVELET_CONV
#define CHECK_CUDA(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        mexErrMsgIdAndTxt("CUDA:Runtime", "%s: %s", msg, cudaGetErrorString(err)); \
    } \
}
#else
#define CHECK_CUDA(msg)
#endif

// ===============
// Kernel
// ===============
__global__ void conv2d_downsample_kernel(
    const float* input, int in_w, int in_h,
    float* output, int out_w, int out_h,
    const float* row_filter, int row_len,
    const float* col_filter, int col_len)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_w || oy >= out_h) return;

    int ix = ox * 2;
    int iy = oy * 2;
    float sum = 0.0f;

    for (int ky = 0; ky < col_len; ++ky) {
        int y = iy + ky - col_len / 2;
        y = max(0, min(in_h - 1, y));
        for (int kx = 0; kx < row_len; ++kx) {
            int x = ix + kx - row_len / 2;
            x = max(0, min(in_w - 1, x));

            float val = input[y * in_w + x];
            sum += val * row_filter[kx] * col_filter[ky];
        }
    }

    output[oy * out_w + ox] = sum;

#ifdef DEBUG_WAVELET_CONV
    if (ox == 0 && oy == 0) {
        printf("conv2d_downsample_kernel executed for output %d x %d\n", out_w, out_h);
    }
#endif
}

// ===============
// Launcher
// ===============
void convolve2d_downsample(
    const float* d_input, int in_w, int in_h,
    const float* row_filter, int row_len,
    const float* col_filter, int col_len,
    float* d_output, int out_w, int out_h)
{
    dim3 block(TILE, TILE);
    dim3 grid((out_w + TILE - 1) / TILE, (out_h + TILE - 1) / TILE);

#ifdef DEBUG_WAVELET_CONV
    cudaMemset(d_output, 0, sizeof(float) * out_w * out_h);
    CHECK_CUDA("memset output");
#endif

    conv2d_downsample_kernel<<<grid, block>>>(
        d_input, in_w, in_h,
        d_output, out_w, out_h,
        row_filter, row_len,
        col_filter, col_len
    );

#ifdef DEBUG_WAVELET_CONV
    cudaDeviceSynchronize();
    CHECK_CUDA("conv2d_downsample_kernel");
#endif
}

// ===============
// Decomposition Function
// ===============
size_t perform_wavelet_decomposition(
    const float* d_input, int width, int height, int levels,
    const float* filt_lo, size_t len_lo,
    const float* filt_hi, size_t len_hi,
    float*& d_C, std::vector<int2>& S_out,
    std::vector<float*>& subbands_out)
{
    std::vector<float*> LL_per_level;
    std::vector<float*> all_subbands;

    int cur_w = width;
    int cur_h = height;
    const float* d_cur = d_input;
    size_t total_coeffs = 0;

    for (int l = 0; l < levels; ++l) {
        int out_w = cur_w / 2;
        int out_h = cur_h / 2;
        size_t sz = out_w * out_h;

        if (out_w < (int)len_lo || out_h < (int)len_lo) {
            mexErrMsgIdAndTxt("CUDA:Wavelet", "Level %d too small for filter length %zu", l, len_lo);
        }

        float *d_LL, *d_LH, *d_HL, *d_HH;
        cudaMalloc(&d_LL, sz * sizeof(float)); CHECK_CUDA("malloc d_LL");
        cudaMalloc(&d_LH, sz * sizeof(float)); CHECK_CUDA("malloc d_LH");
        cudaMalloc(&d_HL, sz * sizeof(float)); CHECK_CUDA("malloc d_HL");
        cudaMalloc(&d_HH, sz * sizeof(float)); CHECK_CUDA("malloc d_HH");

        convolve2d_downsample(d_cur, cur_w, cur_h, filt_lo, len_lo, filt_lo, len_lo, d_LL, out_w, out_h);
        convolve2d_downsample(d_cur, cur_w, cur_h, filt_lo, len_lo, filt_hi, len_hi, d_LH, out_w, out_h);
        convolve2d_downsample(d_cur, cur_w, cur_h, filt_hi, len_hi, filt_lo, len_lo, d_HL, out_w, out_h);
        convolve2d_downsample(d_cur, cur_w, cur_h, filt_hi, len_hi, filt_hi, len_hi, d_HH, out_w, out_h);

        all_subbands.push_back(d_LH);
        all_subbands.push_back(d_HL);
        all_subbands.push_back(d_HH);

        if (l < levels - 1) {
            d_cur = d_LL;
            cur_w = out_w;
            cur_h = out_h;
        }

        LL_per_level.push_back(d_LL);
        S_out.push_back({out_h, out_w});
        total_coeffs += sz * 3;
    }

    S_out.insert(S_out.begin(), {cur_h, cur_w});
    total_coeffs += cur_h * cur_w;

    cudaMalloc(&d_C, total_coeffs * sizeof(float)); CHECK_CUDA("malloc d_C");
    float* d_ptr = d_C;

    cudaMemcpy(d_ptr, LL_per_level.back(), cur_h * cur_w * sizeof(float), cudaMemcpyDeviceToDevice);
    CHECK_CUDA("copy LL");
    d_ptr += cur_h * cur_w;

    for (size_t i = 0; i < all_subbands.size(); ++i) {
        int2 sz = S_out[i + 1];
        size_t count = sz.x * sz.y;
        cudaMemcpy(d_ptr, all_subbands[i], count * sizeof(float), cudaMemcpyDeviceToDevice);
        CHECK_CUDA("copy detail subband");
        d_ptr += count;
    }

    for (auto p : all_subbands) cudaFree(p);
    for (auto p : LL_per_level) cudaFree(p);

    return total_coeffs;
}

#endif // CONVOLVE2D_DOWNSAMPLE_CUH
