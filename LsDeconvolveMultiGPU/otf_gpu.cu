/*==============================================================================
  otf_gpu_mex.cu
  ------------------------------------------------------------------------------
  Compute 3-D Optical Transfer Function (OTF) on the GPU, optimized.

  Usage:
    otf = otf_gpu_mex(psf, [nx ny nz]);
    otf = otf_gpu_mex(psf, [nx ny nz], scratch);

  Inputs:
    psf        : 3-D unshifted PSF (Y×X×Z)   single gpuArray
    fft_shape  : [nx ny nz] (double)         desired FFT size
    scratch    : (optional) gpuArray         complex single, [nx ny nz]

  Output:
    otf        : 3-D complex single gpuArray (C-order on device)
==============================================================================*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "CUDA error %s:%d: %s", \
                          __FILE__, __LINE__, cudaGetErrorString(err))

#define CUFFT_CHECK(err) \
    if ((err) != CUFFT_SUCCESS) \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUFFT", "cuFFT error %s:%d: %d", \
                          __FILE__, __LINE__, int(err))

// Fused padding + ifftshift kernel
__global__ void pad_ifftshift_kernel(
    const float *src, size_t sx, size_t sy, size_t sz,
    float2 *dst, size_t dx, size_t dy, size_t dz,
    ptrdiff_t pre_x, ptrdiff_t pre_y, ptrdiff_t pre_z)
{
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dx || y >= dy || z >= dz) return;

    // 3D ifftshift indices
    auto shift = [](int i, int d) { int s = d/2; int j = i + s; return (j >= d) ? j - d : j; };
    int xs = shift(x, dx), ys = shift(y, dy), zs = shift(z, dz);

    ptrdiff_t sx_i = ptrdiff_t(xs) - pre_x;
    ptrdiff_t sy_i = ptrdiff_t(ys) - pre_y;
    ptrdiff_t sz_i = ptrdiff_t(zs) - pre_z;

    size_t dst_idx = x + dx * (y + dy * z);

    if (sx_i >= 0 && sx_i < ptrdiff_t(sx) &&
        sy_i >= 0 && sy_i < ptrdiff_t(sy) &&
        sz_i >= 0 && sz_i < ptrdiff_t(sz)) {
        size_t src_idx = size_t(sx_i) + sx * (size_t(sy_i) + sy * size_t(sz_i));
        dst[dst_idx].x = src[src_idx];
        dst[dst_idx].y = 0.f;
    } else {
        dst[dst_idx].x = 0.f;
        dst[dst_idx].y = 0.f;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:nrhs", "otf = otf_gpu_mex(psf, fft_shape[, scratch]);");
    if (nlhs < 1)
        mexErrMsgIdAndTxt("otf_gpu_mex:nlhs", "One output (otf) required.");

    mxInitGPU();

    // PSF
    const mxGPUArray *psf = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(psf) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(psf) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:psf", "psf must be 3-D single gpuArray.");
    const mwSize *psf_dims = mxGPUGetDimensions(psf);
    size_t sx = psf_dims[0], sy = psf_dims[1], sz = psf_dims[2];
    const float *d_psf = static_cast<const float*>(mxGPUGetDataReadOnly(psf));

    // FFT shape
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be [nx ny nz] double.");
    const double *sh = mxGetPr(prhs[1]);
    size_t dx = size_t(sh[0]), dy = size_t(sh[1]), dz = size_t(sh[2]);
    if (!dx || !dy || !dz)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be all positive.");
    mwSize otf_dims[3] = { mwSize(dx), mwSize(dy), mwSize(dz) };

    // Optional: scratch buffer
    mxGPUArray *otf = nullptr;
    bool user_scratch = false;
    if (nrhs == 3 && !mxIsEmpty(prhs[2])) {
        const mxGPUArray *scratch = mxGPUCreateFromMxArray(prhs[2]);
        if (mxGPUGetClassID(scratch) != mxSINGLE_CLASS ||
            !mxGPUGetComplexity(scratch) ||
            mxGPUGetNumberOfDimensions(scratch) != 3)
        {
            mxGPUDestroyGPUArray(const_cast<mxGPUArray*>(scratch));
            mexErrMsgIdAndTxt("otf_gpu_mex:scratch", "scratch must be 3-D complex single gpuArray.");
        }
        const mwSize *sd = mxGPUGetDimensions(scratch);
        if (sd[0] != dx || sd[1] != dy || sd[2] != dz) {
            mxGPUDestroyGPUArray(const_cast<mxGPUArray*>(scratch));
            mexErrMsgIdAndTxt("otf_gpu_mex:scratch", "scratch shape must match [nx ny nz].");
        }
        otf = const_cast<mxGPUArray*>(scratch);
        user_scratch = true;
    } else {
        otf = mxGPUCreateGPUArray(3, otf_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
        user_scratch = false;
    }
    float2 *d_otf = static_cast<float2*>(mxGPUGetData(otf));

    // Fused pad + ifftshift kernel
    ptrdiff_t pre_x = (ptrdiff_t)((dx - sx) / 2);
    ptrdiff_t pre_y = (ptrdiff_t)((dy - sy) / 2);
    ptrdiff_t pre_z = (ptrdiff_t)((dz - sz) / 2);

    dim3 blk(16, 8, 4);
    dim3 grd((dz+blk.x-1)/blk.x, (dy+blk.y-1)/blk.y, (dx+blk.z-1)/blk.z);
    pad_ifftshift_kernel<<<grd, blk>>>(
        d_psf, sx, sy, sz, d_otf, dx, dy, dz, pre_x, pre_y, pre_z
    );
    CUDA_CHECK(cudaGetLastError());

    // === Persistent FFT plan (caches for one size) ===
    static cufftHandle cached_plan = 0;
    static size_t cached_dx=0, cached_dy=0, cached_dz=0;
    if (!cached_plan || cached_dx != dx || cached_dy != dy || cached_dz != dz) {
        if (cached_plan) cufftDestroy(cached_plan);
        CUFFT_CHECK(cufftPlan3d(&cached_plan, (int)dz, (int)dy, (int)dx, CUFFT_C2C));
        cached_dx = dx; cached_dy = dy; cached_dz = dz;
    }
    cufftHandle plan = cached_plan;
    CUFFT_CHECK(cufftExecC2C(plan,
        reinterpret_cast<cufftComplex*>(d_otf),
        reinterpret_cast<cufftComplex*>(d_otf),
        CUFFT_FORWARD));

    // No device sync (let MATLAB sync only after all ops done)
    plhs[0] = mxGPUCreateMxArrayOnGPU(otf);
    mxGPUDestroyGPUArray(psf);
    if (!user_scratch) mxGPUDestroyGPUArray(otf);
}
