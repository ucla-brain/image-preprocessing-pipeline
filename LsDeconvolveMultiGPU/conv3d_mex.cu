// File: conv3d_mex.cu
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

__global__ void conv3d_single(
    const float* img, const float* kernel, float* out,
    int nx, int ny, int nz, int kx, int ky, int kz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int kx2 = kx / 2, ky2 = ky / 2, kz2 = kz / 2;
    float acc = 0.0f;
    for (int dz = 0; dz < kz; dz++) {
        int zz = min(max(z + dz - kz2, 0), nz-1);
        for (int dy = 0; dy < ky; dy++) {
            int yy = min(max(y + dy - ky2, 0), ny-1);
            for (int dx = 0; dx < kx; dx++) {
                int xx = min(max(x + dx - kx2, 0), nx-1);
                acc += img[zz*nx*ny + yy*nx + xx] * kernel[dz*kx*ky + dy*kx + dx];
            }
        }
    }
    out[z*nx*ny + y*nx + x] = acc;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxInitGPU();

    // Only supports single, 3D, gpuArray inputs!
    const mxGPUArray *img = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *ker = mxGPUCreateFromMxArray(prhs[1]);
    if (mxGPUGetClassID(img) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(img) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Input", "Input must be 3D gpuArray single.");
    if (mxGPUGetClassID(ker) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(ker) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Kernel", "Kernel must be 3D gpuArray single.");

    const mwSize *isz = mxGPUGetDimensions(img);
    const mwSize *ksz = mxGPUGetDimensions(ker);

    int nx = isz[0], ny = isz[1], nz = isz[2];
    int kx = ksz[0], ky = ksz[1], kz = ksz[2];

    const float *d_img = (const float*)mxGPUGetDataReadOnly(img);
    const float *d_ker = (const float*)mxGPUGetDataReadOnly(ker);

    mxGPUArray *out = mxGPUCreateGPUArray(3, isz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float *d_out = (float*)mxGPUGetData(out);

    dim3 block(8, 8, 4);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    conv3d_single<<<grid, block>>>(d_img, d_ker, d_out, nx, ny, nz, kx, ky, kz);
    cudaDeviceSynchronize();

    plhs[0] = mxGPUCreateMxArrayOnGPU(out);

    mxGPUDestroyGPUArray(img);
    mxGPUDestroyGPUArray(ker);
    mxGPUDestroyGPUArray(out);
}
