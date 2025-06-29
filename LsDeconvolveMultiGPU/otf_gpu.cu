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

// Kernel: Zero-padded, centered copy with axis swap (no extra temp buffers)
// Block size is tuned for most recent NVIDIA GPUs
__global__ void pad_center_swap_fast(
    const float *src, size_t sx, size_t sy, size_t sz,
    float2 *dst,       size_t dx, size_t dy, size_t dz,
    ptrdiff_t pre_x, ptrdiff_t pre_y, ptrdiff_t pre_z)
{
    // Each thread writes one output element
    size_t out_z = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t out_x = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x >= dx || out_y >= dy || out_z >= dz) return;

    ptrdiff_t sx_i = ptrdiff_t(out_x) - pre_x;
    ptrdiff_t sy_i = ptrdiff_t(out_y) - pre_y;
    ptrdiff_t sz_i = ptrdiff_t(out_z) - pre_z;

    size_t dst_idx = out_x + dx * (out_y + dy * out_z); // C-order for cuFFT

    if (sx_i >= 0 && sx_i < ptrdiff_t(sx) &&
        sy_i >= 0 && sy_i < ptrdiff_t(sy) &&
        sz_i >= 0 && sz_i < ptrdiff_t(sz))
    {
        size_t src_idx = size_t(sx_i) + sx * (size_t(sy_i) + sy * size_t(sz_i));
        dst[dst_idx].x = src[src_idx];
        dst[dst_idx].y = 0.f;
    } else {
        dst[dst_idx].x = 0.f;
        dst[dst_idx].y = 0.f;
    }
}

// In-place full 3D ifftshift (optimized: launch with max block size for occupancy)
__device__ __forceinline__ int ifftshift_i(int i, int dim)
{
    int s = dim / 2;
    int j = i + s;
    return (j >= dim) ? j - dim : j;
}
__global__ void ifftshift3D_fast(float2 *v, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int i2 = ifftshift_i(i, nx);
    int j2 = ifftshift_i(j, ny);
    int k2 = ifftshift_i(k, nz);

    // Only swap if this thread "owns" the pair (each pair swapped once)
    if ( (i  < i2) ||
         (i == i2 && j  < j2) ||
         (i == i2 && j == j2 && k < k2) )
    {
        size_t a = i  + nx * (j  + ny * k );
        size_t b = i2 + nx * (j2 + ny * k2);
        float2 tmp = v[a];
        v[a] = v[b];
        v[b] = tmp;
    }
}

// -----------------------------------------------------------------------------
// MEX entrypoint: Optimized with minimal synchronizations and best CUDA practice
// -----------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:nrhs", "Usage: otf = otf_gpu_mex(psf, fft_shape[, scratch]);");
    if (nlhs < 1)
        mexErrMsgIdAndTxt("otf_gpu_mex:nlhs", "One output (otf) required.");

    mxInitGPU();

    // --- PSF ---
    const mxGPUArray *psf = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(psf) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(psf) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:psf", "psf must be a 3-D single gpuArray.");
    const mwSize *psf_dims = mxGPUGetDimensions(psf);
    size_t sx = psf_dims[0], sy = psf_dims[1], sz = psf_dims[2];
    const float *d_psf = static_cast<const float*>(mxGPUGetDataReadOnly(psf));

    // --- FFT shape ---
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be [nx ny nz] double.");
    const double *sh = mxGetPr(prhs[1]);
    size_t dx = size_t(sh[0]), dy = size_t(sh[1]), dz = size_t(sh[2]);
    if (!dx || !dy || !dz)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be all positive.");
    mwSize otf_dims[3] = { mwSize(dx), mwSize(dy), mwSize(dz) };

    // --- Optional: scratch buffer ---
    mxGPUArray *otf = nullptr;
    bool user_scratch = false;
    if (nrhs == 3 && !mxIsEmpty(prhs[2])) {
        const mxGPUArray *scratch = mxGPUCreateFromMxArray(prhs[2]);
        if (mxGPUGetClassID(scratch) != mxSINGLE_CLASS ||
            !mxGPUGetComplexity(scratch) ||
            mxGPUGetNumberOfDimensions(scratch) != 3)
        {
            mxGPUDestroyGPUArray(const_cast<mxGPUArray*>(scratch));
            mexErrMsgIdAndTxt("otf_gpu_mex:scratch",
                "scratch must be a 3-D complex single gpuArray.");
        }
        const mwSize *sd = mxGPUGetDimensions(scratch);
        if (sd[0] != dx || sd[1] != dy || sd[2] != dz) {
            mxGPUDestroyGPUArray(const_cast<mxGPUArray*>(scratch));
            mexErrMsgIdAndTxt("otf_gpu_mex:scratch",
                "scratch buffer shape must match [nx ny nz].");
        }
        otf = const_cast<mxGPUArray*>(scratch);
        user_scratch = true;
    } else {
        otf = mxGPUCreateGPUArray(3, otf_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
        user_scratch = false;
    }
    float2 *d_otf = static_cast<float2*>(mxGPUGetData(otf));

    // --- Padding (fewer but larger blocks for higher occupancy) ---
    // Use 16 threads per dimension for better GPU occupancy (512 threads/block)
    dim3 blk(16, 8, 4);
    dim3 grd((dz+blk.x-1)/blk.x, (dy+blk.y-1)/blk.y, (dx+blk.z-1)/blk.z);

    ptrdiff_t pre_x = (ptrdiff_t)((dx - sx) / 2);
    ptrdiff_t pre_y = (ptrdiff_t)((dy - sy) / 2);
    ptrdiff_t pre_z = (ptrdiff_t)((dz - sz) / 2);

    pad_center_swap_fast<<<grd, blk>>>(d_psf, sx, sy, sz, d_otf, dx, dy, dz, pre_x, pre_y, pre_z);
    CUDA_CHECK(cudaGetLastError());

    // --- In-place ifftshift (single kernel, as big as possible) ---
    dim3 blk2(8, 8, 8);
    dim3 grd2((dx+blk2.x-1)/blk2.x, (dy+blk2.y-1)/blk2.y, (dz+blk2.z-1)/blk2.z);

    ifftshift3D_fast<<<grd2, blk2>>>(d_otf, (int)dx, (int)dy, (int)dz);
    CUDA_CHECK(cudaGetLastError());

    // --- cuFFT: Only one plan per call, in-place, no sync ---
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, (int)dz, (int)dy, (int)dx, CUFFT_C2C));
    CUFFT_CHECK(cufftExecC2C(plan,
        reinterpret_cast<cufftComplex*>(d_otf),
        reinterpret_cast<cufftComplex*>(d_otf),
        CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));

    // Don't call cudaDeviceSynchronize(); let MATLAB handle this for minimal latency

    plhs[0] = mxGPUCreateMxArrayOnGPU(otf);
    mxGPUDestroyGPUArray(psf);
    if (!user_scratch) {
        mxGPUDestroyGPUArray(otf);
    }
}
