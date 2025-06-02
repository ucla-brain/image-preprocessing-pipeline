#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>

// Pad/crop kernel: sets out-of-bounds to zero, preserves input region
__global__ void zero_pad_crop_kernel(
    const float* src, int sx, int sy, int sz,
    float2* dst, int dx, int dy, int dz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dx && y < dy && z < dz) {
        int dst_idx = x + dx * (y + dy * z);
        if (x < sx && y < sy && z < sz) {
            int src_idx = x + sx * (y + sy * z);
            dst[dst_idx].x = src[src_idx]; // real
            dst[dst_idx].y = 0.0f;         // imag
        } else {
            dst[dst_idx].x = 0.0f;
            dst[dst_idx].y = 0.0f;
        }
    }
}

// Compute conjugate (complex numbers)
__global__ void conjugate_kernel(const float2* src, float2* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx].x = src[idx].x;
        dst[idx].y = -src[idx].y;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // ---- Boilerplate checks ----
    if (nrhs != 2) mexErrMsgIdAndTxt("calculate_otf_mex:nrhs", "2 inputs required: psf, fft_shape.");
    if (nlhs < 2) mexErrMsgIdAndTxt("calculate_otf_mex:nlhs", "2 outputs required: otf, otf_conj.");

    mxInitGPU();

    // ---- Input: PSF (3D, single, gpuArray) ----
    const mxGPUArray* psf = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(psf) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("calculate_otf_mex:psfType", "Input psf must be single.");
    if (mxGPUGetNumberOfDimensions(psf) != 3)
        mexErrMsgIdAndTxt("calculate_otf_mex:psfDims", "Input psf must be 3D.");

    const mwSize* psf_dims = mxGPUGetDimensions(psf);
    int sx = static_cast<int>(psf_dims[0]);
    int sy = static_cast<int>(psf_dims[1]);
    int sz = static_cast<int>(psf_dims[2]);
    const float* d_psf = static_cast<const float*>(mxGPUGetDataReadOnly(psf));

    // ---- Input: fft_shape ([3] numeric array) ----
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 3)
        mexErrMsgIdAndTxt("calculate_otf_mex:fftShape", "fft_shape must be [nx ny nz] double array.");

    double* fft_shape = mxGetPr(prhs[1]);
    int dx = static_cast<int>(fft_shape[0]);
    int dy = static_cast<int>(fft_shape[1]);
    int dz = static_cast<int>(fft_shape[2]);

    mwSize out_dims[3] = {static_cast<mwSize>(dx), static_cast<mwSize>(dy), static_cast<mwSize>(dz)};
    int N = dx * dy * dz;

    // ---- Allocate output buffer (single, complex, gpuArray) ----
    mxGPUArray* otf_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf = static_cast<float2*>(mxGPUGetData(otf_gpu));

    // ---- Pad/crop input into output buffer ----
    dim3 block(8,8,8), grid((dx+7)/8, (dy+7)/8, (dz+7)/8);
    zero_pad_crop_kernel<<<grid, block>>>(d_psf, sx, sy, sz, d_otf, dx, dy, dz);
    cudaDeviceSynchronize();

    // ---- cuFFT: 3D FFT (in-place) ----
    cufftHandle plan;
    cufftPlan3d(&plan, dx, dy, dz, CUFFT_C2C); // single complex-to-complex
    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_otf), reinterpret_cast<cufftComplex*>(d_otf), CUFFT_FORWARD);
    cufftDestroy(plan);

    // ---- Allocate buffer for conjugate ----
    mxGPUArray* otf_conj_gpu = mxGPUCreateGPUArray(3, out_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    float2* d_otf_conj = static_cast<float2*>(mxGPUGetData(otf_conj_gpu));

    // ---- Compute conjugate ----
    conjugate_kernel<<<(N+255)/256, 256>>>(d_otf, d_otf_conj, N);
    cudaDeviceSynchronize();

    // ---- Return to MATLAB as gpuArray ----
    plhs[0] = mxGPUCreateMxArrayOnGPU(otf_gpu);
    plhs[1] = mxGPUCreateMxArrayOnGPU(otf_conj_gpu);

    // ---- Cleanup ----
    mxGPUDestroyGPUArray(psf);
    mxGPUDestroyGPUArray(otf_gpu);
    mxGPUDestroyGPUArray(otf_conj_gpu);
}
