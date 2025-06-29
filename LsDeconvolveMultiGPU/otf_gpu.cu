/*==============================================================================
  otf_gpu_mex.cu
  ------------------------------------------------------------------------------
  Compute 3-D Optical Transfer Function (OTF) on the GPU, using a user-provided
  complex gpuArray as buffer and output.

  Usage in MATLAB (all gpuArray, single):
      otf = otf_gpu_mex(psf, [nx ny nz], ..., buffer);

  Inputs
  ──────
    psf        : 3-D unshifted PSF (Y×X×Z)       single gpuArray, real
    fft_shape  : [nx ny nz]                      double, output size
    buffer     : 3-D single complex gpuArray     used as internal + output buffer

  Output
  ──────
    otf        : 3-D complex single gpuArray     (buffer, filled with OTF)
==============================================================================*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>

// ──────────────── Error-handling helpers ────────────────
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUDA", "CUDA error %s:%d: %s", \
                          __FILE__, __LINE__, cudaGetErrorString(err));

#define CUFFT_CHECK(err) \
    if ((err) != CUFFT_SUCCESS) \
        mexErrMsgIdAndTxt("otf_gpu_mex:CUFFT", "cuFFT error %s:%d: %d", \
                          __FILE__, __LINE__, int(err));

// ──────────────── Kernel: 0-filled, centred pad + axis swap ────────────────
__global__ void pad_center_swap(
    const float *src, size_t sx, size_t sy, size_t sz,
    float2 *dst,       size_t dx, size_t dy, size_t dz,
    ptrdiff_t pre_x, ptrdiff_t pre_y, ptrdiff_t pre_z)
{
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;  // NOTE: Z fastest for cuFFT
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dx || y >= dy || z >= dz) return;

    ptrdiff_t sx_i = ptrdiff_t(x) - pre_x;
    ptrdiff_t sy_i = ptrdiff_t(y) - pre_y;
    ptrdiff_t sz_i = ptrdiff_t(z) - pre_z;

    size_t dst_idx = x + dx * (y + dy * z); // C-order

    if (sx_i >= 0 && sx_i < ptrdiff_t(sx) &&
        sy_i >= 0 && sy_i < ptrdiff_t(sy) &&
        sz_i >= 0 && sz_i < ptrdiff_t(sz))
    {
        size_t src_idx = size_t(sx_i) + sx * (size_t(sy_i) + sy * size_t(sz_i));
        dst[dst_idx].x = src[src_idx];
        dst[dst_idx].y = 0.f;
    }
    else
    {
        dst[dst_idx].x = 0.f;
        dst[dst_idx].y = 0.f;
    }
}

// ──────────────── Kernel: full 3-D ifftshift ────────────────
__device__ __forceinline__ int ifftshift_i(int i, int dim)
{
    int s = dim / 2;
    int j = i + s;
    return (j >= dim) ? j - dim : j;
}

__global__ void ifftshift3D(float2 *v, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int i2 = ifftshift_i(i, nx);
    int j2 = ifftshift_i(j, ny);
    int k2 = ifftshift_i(k, nz);

    // swap once per pair
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

// ───────────────────────── MEX entry ─────────────────────────
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // 4th argument required: user-provided buffer
    if (nrhs != 3 && nrhs != 4)
        mexErrMsgIdAndTxt("otf_gpu_mex:nrhs", "Three or four inputs required (psf, fft_shape, [unused], buffer).");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("otf_gpu_mex:nlhs", "One output (otf) required.");

    mxInitGPU();

    // ---- PSF ----
    const mxGPUArray *psf = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(psf) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(psf) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:psf", "psf must be 3-D single gpuArray.");

    const mwSize *pd = mxGPUGetDimensions(psf);
    size_t sx = pd[0], sy = pd[1], sz = pd[2];
    const float *d_psf = static_cast<const float*>(mxGPUGetDataReadOnly(psf));

    // ---- fft_shape ----
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 3)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be [nx ny nz] double.");

    double *sh = mxGetPr(prhs[1]);
    size_t dx = size_t(sh[0]), dy = size_t(sh[1]), dz = size_t(sh[2]);
    if (!dx || !dy || !dz)
        mexErrMsgIdAndTxt("otf_gpu_mex:fftshape", "fft_shape must be positive.");

    mwSize odims[3] = { mwSize(dx), mwSize(dy), mwSize(dz) };

    // ---- User-provided buffer ----
    if (nrhs < 4)
        mexErrMsgIdAndTxt("otf_gpu_mex:buffer", "User-provided buffer (complex single gpuArray) required as 4th argument.");

    mxGPUArray *user_buffer = const_cast<mxGPUArray*>(mxGPUCreateFromMxArray(prhs[3]));
    if (mxGPUGetClassID(user_buffer) != mxSINGLE_CLASS ||
        !mxGPUGetIsComplex(user_buffer) ||
        mxGPUGetNumberOfDimensions(user_buffer) != 3)
    {
        mxGPUDestroyGPUArray(user_buffer);
        mxGPUDestroyGPUArray(psf);
        mexErrMsgIdAndTxt("otf_gpu_mex:buffer", "Buffer must be 3-D complex single gpuArray.");
    }

    const mwSize *bdims = mxGPUGetDimensions(user_buffer);
    if (bdims[0] != dx || bdims[1] != dy || bdims[2] != dz)
    {
        mxGPUDestroyGPUArray(user_buffer);
        mxGPUDestroyGPUArray(psf);
        mexErrMsgIdAndTxt("otf_gpu_mex:buffer", "Buffer must match fft_shape.");
    }

    float2 *d_otf = static_cast<float2*>(mxGPUGetData(user_buffer));

    // ---- Zero-pad & centre PSF into buffer ----
    dim3 blk(8,8,8);
    dim3 grd( (dz+blk.x-1)/blk.x,
              (dy+blk.y-1)/blk.y,
              (dx+blk.z-1)/blk.z );
    ptrdiff_t pre_x = (ptrdiff_t)( (dx - sx) / 2 );
    ptrdiff_t pre_y = (ptrdiff_t)( (dy - sy) / 2 );
    ptrdiff_t pre_z = (ptrdiff_t)( (dz - sz) / 2 );

    pad_center_swap<<<grd, blk>>>(d_psf, sx, sy, sz,
                                  d_otf, dx, dy, dz,
                                  pre_x, pre_y, pre_z);
    CUDA_CHECK(cudaGetLastError());

    // ---- ifftshift in place (buffer) ----
    dim3 grd2( (dx+blk.x-1)/blk.x,
               (dy+blk.y-1)/blk.y,
               (dz+blk.z-1)/blk.z );
    ifftshift3D<<<grd2, blk>>>(d_otf, (int)dx, (int)dy, (int)dz);
    CUDA_CHECK(cudaGetLastError());

    // ---- 3-D FFT in place (buffer) ----
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d((int)dz, (int)dy, (int)dx, CUFFT_C2C));
    CUFFT_CHECK(cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex*>(d_otf),
                 reinterpret_cast<cufftComplex*>(d_otf),
                 CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan));

    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Return the user-provided buffer as the OTF (MATLAB gpuArray) ----
    plhs[0] = mxGPUCreateMxArrayOnGPU(user_buffer);

    // ---- Free input and local references (but not the user buffer, which is output) ----
    mxGPUDestroyGPUArray(psf);
    mxGPUDestroyGPUArray(user_buffer); // safe to destroy here, as output is a new mxArray referencing the same GPU data
}
