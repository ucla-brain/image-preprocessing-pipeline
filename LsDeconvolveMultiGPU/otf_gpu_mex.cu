#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cassert>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    }

#define CUFFT_CHECK(err) \
    if (err != CUFFT_SUCCESS) { \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUFFT", "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, err); \
    }

// Kernel for centered zero-padding and cropping
__global__ void zero_pad_crop_centered(
    const float* src, size_t sx, size_t sy, size_t sz,
    float2* dst, size_t dx, size_t dy, size_t dz,
    ptrdiff_t pre_x, ptrdiff_t pre_y, ptrdiff_t pre_z)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dx && y < dy && z < dz) {
        ptrdiff_t src_x = (ptrdiff_t)x - pre_x;
        ptrdiff_t src_y = (ptrdiff_t)y - pre_y;
        ptrdiff_t src_z = (ptrdiff_t)z - pre_z;
        size_t dst_idx = x + dx * (y + dy * z);
        if (src_x >= 0 && (size_t)src_x < sx &&
            src_y >= 0 && (size_t)src_y < sy &&
            src_z >= 0 && (size_t)src_z < sz) {
            size_t src_idx = (size_t)src_x + sx * ((size_t)src_y + sy * (size_t)src_z);
            dst[dst_idx].x = src[src_idx]; // real part
            dst[dst_idx].y = 0.0f;         // imag part
        } else {
            dst[dst_idx].x = 0.0f;
            dst[dst_idx].y = 0.0f;
        }
    }
}

// Kernel for conjugating a complex array
__global__ void conjugate_kernel(const float2* src, float2* dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx].x = src[idx].x;
        dst[idx].y = -src[idx].y;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // ==== Input checks ====
    if (nrhs != 2)
        mexErrMsgIdAndTxt("otf_gpu_mex:nrhs", "2 inputs required: psf, fft_shape.");
    if (nlhs < 2)
        mexErrMsgIdAndTxt("otf_gpu_mex:nlhs", "2 outputs required: otf, otf_conj.");

    mxInitGPU();

    // ---- Input: PSF (3D, single, gpuArray) ----
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

    // ---- Input: fft_shape ([3] double) ----
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

    // ---- Allocate output buffer (single, complex, gpuArray) ----
    mxGPUArray* otf_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf = static_cast<float2*>(mxGPUGetData(otf_gpu));

    // ---- Pad/crop input into output buffer ----
    dim3 block(8,8,8), grid((dx+7)/8, (dy+7)/8, (dz+7)/8);
    zero_pad_crop_centered<<<grid, block>>>(
        d_psf, sx, sy, sz, d_otf, dx, dy, dz, prepad_x, prepad_y, prepad_z
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- cuFFT: 3D FFT (in-place) ----
    cufftHandle plan;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, (int)dx, (int)dy, (int)dz, CUFFT_C2C));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_otf), reinterpret_cast<cufftComplex*>(d_otf), CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Allocate buffer for conjugate ----
    mxGPUArray* otf_conj_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf_conj = static_cast<float2*>(mxGPUGetData(otf_conj_gpu));

    // ---- Compute conjugate ----
    conjugate_kernel<<<(N+255)/256, 256>>>(d_otf, d_otf_conj, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Synchronize one more time before returning to MATLAB ----
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Return to MATLAB as gpuArray ----
    plhs[0] = mxGPUCreateMxArrayOnGPU(otf_gpu);
    plhs[1] = mxGPUCreateMxArrayOnGPU(otf_conj_gpu);

    // ---- Cleanup ----
    mxGPUDestroyGPUArray(psf);
}
