/*==============================================================================
  conj_gpu.cu
  ------------------------------------------------------------------------------
  Compute the complex conjugate of an existing single-precision OTF on the GPU.

  Usage in MATLAB (gpuArray):
      otf_conj = conj_gpu(otf);

  Input
  ──────
    otf         : 3-D complex single gpuArray (output of otf_gpu)

  Output
  ──────
    otf_conj    : 3-D complex single gpuArray  (same size as input)
==============================================================================*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

// ─────────────── Error macro ───────────────
#define CUDA_CHECK(e) \
    if ((e) != cudaSuccess) \
        mexErrMsgIdAndTxt("conj_gpu:CUDA", "CUDA error %s:%d: %s", \
                          __FILE__, __LINE__, cudaGetErrorString(e));

// ─────────────── Kernel ───────────────
__global__ void conj_kernel(const float2 *src, float2 *dst, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        dst[idx].x = src[idx].x;
        dst[idx].y = -src[idx].y;
    }
}

// ─────────────── MEX entry ───────────────
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgIdAndTxt("conj_gpu:nrhs", "One input (otf) required.");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("conj_gpu:nlhs", "One output (otf_conj) required.");

    mxInitGPU();

    const mxGPUArray *otf = mxGPUCreateFromMxArray(prhs[0]);

    if (mxGPUGetClassID(otf) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("conj_gpu:type", "Input must be single precision.");
    if (!mxGPUIsComplex(otf))
        mexErrMsgIdAndTxt("conj_gpu:complex", "Input must be complex.");

    // Compute element count manually
    size_t N = 1;
    const mwSize *dims = mxGPUGetDimensions(otf);
    mwSize nd = mxGPUGetNumberOfDimensions(otf);
    for (mwSize i = 0; i < nd; ++i) N *= dims[i];

    const float2 *d_in = static_cast<const float2*>(mxGPUGetDataReadOnly(otf));

    mxGPUArray *out = mxGPUCreateGPUArray(nd, dims,
                                          mxSINGLE_CLASS, mxCOMPLEX,
                                          MX_GPU_DO_NOT_INITIALIZE);
    float2 *d_out = static_cast<float2*>(mxGPUGetData(out));

    dim3 blk(256);
    dim3 grd((N + blk.x - 1) / blk.x);
    conj_kernel<<<grd, blk>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    mxGPUDestroyGPUArray(otf);
}
