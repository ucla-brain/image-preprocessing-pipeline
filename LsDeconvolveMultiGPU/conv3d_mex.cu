// conv3d_mex.cu -- Optimized 3D convolution with replicate boundary for MATLAB GPU

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
        mexErrMsgIdAndTxt("conv3d_mex:CUDA", "CUDA error %s:%d: %s", file, line, cudaGetErrorString(code));
}

__global__ void conv3d_kernel(
    const float* __restrict__ img, const float* __restrict__ kernel, float* __restrict__ out,
    int nx, int ny, int nz, int kx, int ky, int kz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    // Kernel center for flip
    int kx2 = kx / 2, ky2 = ky / 2, kz2 = kz / 2;

    float acc = 0.0f;
    #pragma unroll 1
    for (int dz = 0; dz < kz; dz++) {
        int iz = z + dz - kz2;
        iz = max(0, min(iz, nz-1)); // replicate boundary
        #pragma unroll 1
        for (int dy = 0; dy < ky; dy++) {
            int iy = y + dy - ky2;
            iy = max(0, min(iy, ny-1));
            #pragma unroll 1
            for (int dx = 0; dx < kx; dx++) {
                int ix = x + dx - kx2;
                ix = max(0, min(ix, nx-1));
                // Flip kernel for convolution:
                int fx = kx-1-dx, fy = ky-1-dy, fz = kz-1-dz;
                acc += img[ix + iy*nx + iz*nx*ny] * kernel[fx + fy*kx + fz*kx*ky];
            }
        }
    }
    out[x + y*nx + z*nx*ny] = acc;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxInitGPU();

    if (nrhs != 2)
        mexErrMsgIdAndTxt("conv3d_mex:Args", "Requires two inputs: img, kernel (both gpuArray single 3D).");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("conv3d_mex:Args", "One output only.");

    // Validate and get GPU arrays
    const mxGPUArray *img = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *ker = mxGPUCreateFromMxArray(prhs[1]);
    if (mxGPUGetClassID(img) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(img) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Input", "Input must be 3D gpuArray single.");
    if (mxGPUGetClassID(ker) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(ker) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Kernel", "Kernel must be 3D gpuArray single.");

    const mwSize *isz = mxGPUGetDimensions(img);
    const mwSize *ksz = mxGPUGetDimensions(ker);

    int nx = static_cast<int>(isz[0]), ny = static_cast<int>(isz[1]), nz = static_cast<int>(isz[2]);
    int kx = static_cast<int>(ksz[0]), ky = static_cast<int>(ksz[1]), kz = static_cast<int>(ksz[2]);

    const float *d_img = static_cast<const float*>(mxGPUGetDataReadOnly(img));
    const float *d_ker = static_cast<const float*>(mxGPUGetDataReadOnly(ker));

    mxGPUArray *out = mxGPUCreateGPUArray(3, isz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float *d_out = static_cast<float*>(mxGPUGetData(out));

    // Choose block size for best occupancy with 3D grids (8x8x4 is generally a sweet spot for modern GPUs)
    const dim3 block(8, 8, 4);
    const dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    conv3d_kernel<<<grid, block>>>(d_img, d_ker, d_out, nx, ny, nz, kx, ky, kz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plhs[0] = mxGPUCreateMxArrayOnGPU(out);

    // Cleanup
    mxGPUDestroyGPUArray(img);
    mxGPUDestroyGPUArray(ker);
    mxGPUDestroyGPUArray(out);
}
