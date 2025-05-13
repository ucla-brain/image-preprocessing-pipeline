// ===============================
// convolve2d_upsample.cuh
// ===============================
#ifndef CONVOLVE2D_UPSAMPLE_CUH
#define CONVOLVE2D_UPSAMPLE_CUH

#include <cuda_runtime.h>
#include <vector>
#include <cassert>

#define TILE 16

// 2D upsample + convolve kernel
__global__ void conv2d_upsample_kernel(
    const float* input, int in_w, int in_h,
    float* output, int out_w, int out_h,
    const float* row_filter, int row_len,
    const float* col_filter, int col_len)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h) return;

    float sum = 0.0f;
    for (int ky = 0; ky < col_len; ++ky) {
        int sy = oy - ky + col_len / 2;
        if (sy % 2 != 0) continue;
        int iy = sy / 2;
        if (iy < 0 || iy >= in_h) continue;
        for (int kx = 0; kx < row_len; ++kx) {
            int sx = ox - kx + row_len / 2;
            if (sx % 2 != 0) continue;
            int ix = sx / 2;
            if (ix < 0 || ix >= in_w) continue;

            float val = input[iy * in_w + ix];
            sum += val * row_filter[kx] * col_filter[ky];
        }
    }
    output[oy * out_w + ox] += sum;
}

// Launch upsample + filter kernel
void convolve2d_upsample_add(
    const float* d_input, int in_w, int in_h,
    const float* row_filter, int row_len,
    const float* col_filter, int col_len,
    float* d_output, int out_w, int out_h)
{
    dim3 block(TILE, TILE);
    dim3 grid((out_w + TILE - 1) / TILE, (out_h + TILE - 1) / TILE);

    conv2d_upsample_kernel<<<grid, block>>>(
        d_input, in_w, in_h,
        d_output, out_w, out_h,
        row_filter, row_len,
        col_filter, col_len);
}

// Top-level function for multilevel reconstruction
void reconstruct_full_image(
    const float* d_C, const int2* d_S, int levels,
    const float* Lo_R, size_t len_lo,
    const float* Hi_R, size_t len_hi,
    float* d_out)
{
    std::vector<float*> LL_stack;
    size_t offset = 0;

    // Initial LL
    int2 LL_size = d_S[0];
    int h = LL_size.x, w = LL_size.y;
    size_t sz_LL = h * w;

    float* d_LL;
    cudaMalloc(&d_LL, sz_LL * sizeof(float));
    cudaMemcpy(d_LL, d_C + offset, sz_LL * sizeof(float), cudaMemcpyDeviceToDevice);
    offset += sz_LL;

    for (int l = 1; l <= levels; ++l) {
        int2 sz = d_S[l];
        int h2 = sz.x;
        int w2 = sz.y;
        size_t sz_sub = h2 * w2;

        float *d_LH, *d_HL, *d_HH;
        cudaMalloc(&d_LH, sz_sub * sizeof(float));
        cudaMalloc(&d_HL, sz_sub * sizeof(float));
        cudaMalloc(&d_HH, sz_sub * sizeof(float));

        cudaMemcpy(d_LH, d_C + offset, sz_sub * sizeof(float), cudaMemcpyDeviceToDevice); offset += sz_sub;
        cudaMemcpy(d_HL, d_C + offset, sz_sub * sizeof(float), cudaMemcpyDeviceToDevice); offset += sz_sub;
        cudaMemcpy(d_HH, d_C + offset, sz_sub * sizeof(float), cudaMemcpyDeviceToDevice); offset += sz_sub;

        int new_h = h2 * 2;
        int new_w = w2 * 2;
        float* d_temp;
        cudaMalloc(&d_temp, new_h * new_w * sizeof(float));
        cudaMemset(d_temp, 0, new_h * new_w * sizeof(float));

        convolve2d_upsample_add(d_LL, w2, h2, Lo_R, len_lo, Lo_R, len_lo, d_temp, new_w, new_h);
        convolve2d_upsample_add(d_LH, w2, h2, Hi_R, len_hi, Lo_R, len_lo, d_temp, new_w, new_h);
        convolve2d_upsample_add(d_HL, w2, h2, Lo_R, len_lo, Hi_R, len_hi, d_temp, new_w, new_h);
        convolve2d_upsample_add(d_HH, w2, h2, Hi_R, len_hi, Hi_R, len_hi, d_temp, new_w, new_h);

        cudaFree(d_LL);  // previous LL
        cudaFree(d_LH);
        cudaFree(d_HL);
        cudaFree(d_HH);

        d_LL = d_temp;
        h = new_h;
        w = new_w;
    }

    cudaMemcpy(d_out, d_LL, sizeof(float) * h * w, cudaMemcpyDeviceToDevice);
    cudaFree(d_LL);
}

#endif // CONVOLVE2D_UPSAMPLE_CUH
