/*
Written by Keivan Moradi code review by ChatGPT 4.1 2025
GPL v3 license

I made the cuda version of otf function in matlab hopping to reduce the ram usage, but in reality its ram usage is
similar to matlab. Speed improvement is significant for large array because of native compilation and other optimizations
*/

// File: otf_gpu_mex.cu

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

// ------------------- Utility Macros -------------------
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    }

#define CUFFT_CHECK(err) \
    if (err != CUFFT_SUCCESS) { \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUFFT", "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, err); \
    }

// ------------------ Zero-pad kernel -------------------
__global__ void zero_pad_crop_centered_swapped(
    const float* src, size_t sx, size_t sy, size_t sz,    // Source dims (MATLAB order)
    float2* dst, size_t dx, size_t dy, size_t dz,         // Dest dims (MATLAB order, but dx/dz swapped for FFT)
    ptrdiff_t pre_x, ptrdiff_t pre_y, ptrdiff_t pre_z)
{
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dx && y < dy && z < dz) {
        ptrdiff_t src_x = (ptrdiff_t)x - pre_x;
        ptrdiff_t src_y = (ptrdiff_t)y - pre_y;
        ptrdiff_t src_z = (ptrdiff_t)z - pre_z;
        size_t dst_idx = x + dx * (y + dy * z);
        if (src_x >= 0 && (size_t)src_x < sx &&
            src_y >= 0 && (size_t)src_y < sy &&
            src_z >= 0 && (size_t)src_z < sz) {
            size_t src_idx = (size_t)src_x + sx * ((size_t)src_y + sy * (size_t)src_z);
            dst[dst_idx].x = src[src_idx];
            dst[dst_idx].y = 0.0f;
        } else {
            dst[dst_idx].x = 0.0f;
            dst[dst_idx].y = 0.0f;
        }
    }
}

// ------------------- General ifftshift kernel -------------------
__device__ int ifftshift_idx(int i, int dim) {
    int shift = dim / 2;
    int idx = i + shift;
    if (idx >= dim) idx -= dim;
    return idx;
}

__global__ void ifftshift3d_full_kernel(float2* data, int nx, int ny, int nz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;

    // Only swap once per pair (diagonal half)
    if (tx < nx && ty < ny && tz < nz) {
        int i0 = tx, j0 = ty, k0 = tz;
        int i1 = ifftshift_idx(i0, nx);
        int j1 = ifftshift_idx(j0, ny);
        int k1 = ifftshift_idx(k0, nz);

        // Only swap if our index is "less than" the shifted one (avoids double swap)
        if ( (i0 < i1) ||
             (i0 == i1 && j0 < j1) ||
             (i0 == i1 && j0 == j1 && k0 < k1) ) {

            size_t idx0 = i0 + nx * (j0 + ny * k0);
            size_t idx1 = i1 + nx * (j1 + ny * k1);

            // Swap
            float2 tmp = data[idx0];
            data[idx0] = data[idx1];
            data[idx1] = tmp;
        }
    }
}

// ------------------- Conjugate kernel -------------------
__global__ void conjugate_kernel(const float2* src, float2* dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx].x = src[idx].x;
        dst[idx].y = -src[idx].y;
    }
}

// ------------------------ MEX ENTRY -------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "2 inputs required: psf, fft_shape.");
    if (nlhs < 2)
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "2 outputs required: otf, otf_conj.");

    mxInitGPU();

    // ---- PSF: 3D, single, gpuArray ----
    const mxGPUArray* psf = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(psf) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("otf_gpu_mex:psfType", "Input psf must be single.");
    if (mxGPUGetNumberOfDimensions(psf) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:psfDims", "Input psf must be 3D.");

    const mwSize* psf_dims = mxGPUGetDimensions(psf);
    size_t sx = static_cast<size_t>(psf_dims[0]);
    size_t sy = static_cast<size_t>(psf_dims[1]);
    size_t sz = static_cast<size_t>(psf_dims[2]);
    const float* d_psf = static_cast<const float*>(mxGPUGetDataReadOnly(psf));

    // ---- fft_shape ----
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftShape", "fft_shape must be [nx ny nz] double array.");
    double* fft_shape = mxGetPr(prhs[1]);
    size_t dx = static_cast<size_t>(fft_shape[0]);
    size_t dy = static_cast<size_t>(fft_shape[1]);
    size_t dz = static_cast<size_t>(fft_shape[2]);

    mwSize out_dims[3] = {static_cast<mwSize>(dx), static_cast<mwSize>(dy), static_cast<mwSize>(dz)};
    size_t N = dx * dy * dz;

    // ---- Centered padding like MATLAB ----
    ptrdiff_t prepad_x = static_cast<ptrdiff_t>((dx - sx)/2);
    ptrdiff_t prepad_y = static_cast<ptrdiff_t>((dy - sy)/2);
    ptrdiff_t prepad_z = static_cast<ptrdiff_t>((dz - sz)/2);

    // ---- Allocate output buffer ----
    mxGPUArray* otf_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf = static_cast<float2*>(mxGPUGetData(otf_gpu));

    // ---- Pad and center ----
    dim3 block(8,8,8), grid((dz+7)/8, (dy+7)/8, (dx+7)/8);
    zero_pad_crop_centered_swapped<<<grid, block>>>(
        d_psf, sx, sy, sz, d_otf, dx, dy, dz, prepad_x, prepad_y, prepad_z
    );
    CUDA_CHECK(cudaGetLastError());

    // ---- ifftshift (axis-wise) for all dims ----
    dim3 block_shift(8,8,8);
    dim3 grid_shift((dx+7)/8, (dy+7)/8, (dz+7)/8);
    ifftshift3d_full_kernel<<<grid_shift, block_shift>>>((float2*)d_otf, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());

    // ---- cuFFT: 3D FFT (dz, dy, dx) ----
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d((int)dz, (int)dy, (int)dx, CUFFT_C2C));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_otf), reinterpret_cast<cufftComplex*>(d_otf), CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));

    // ---- Allocate buffer for conjugate ----
    mxGPUArray* otf_conj_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf_conj = static_cast<float2*>(mxGPUGetData(otf_conj_gpu));

    // ---- Compute conjugate ----
    conjugate_kernel<<<(N+255)/256, 256>>>(d_otf, d_otf_conj, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    plhs[0] = mxGPUCreateMxArrayOnGPU(otf_gpu);
    plhs[1] = mxGPUCreateMxArrayOnGPU(otf_conj_gpu);

    mxGPUDestroyGPUArray(psf);
    mxGPUDestroyGPUArray(otf_gpu);
    mxGPUDestroyGPUArray(otf_conj_gpu);
}
