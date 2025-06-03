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

    int kx2 = kx / 2;
    int ky2 = ky / 2;
    int kz2 = kz / 2;

    float acc = 0.0f;
    for (int dz = 0; dz < kz; dz++) {
        int iz = z + dz - kz2;
        // Clamp for replicate boundary
        iz = iz < 0 ? 0 : (iz >= nz ? nz - 1 : iz);
        for (int dy = 0; dy < ky; dy++) {
            int iy = y + dy - ky2;
            iy = iy < 0 ? 0 : (iy >= ny ? ny - 1 : iy);
            for (int dx = 0; dx < kx; dx++) {
                int ix = x + dx - kx2;
                ix = ix < 0 ? 0 : (ix >= nx ? nx - 1 : ix);

                // MATLAB column-major order for 3D: (ix,iy,iz) = [x,y,z]
                int img_idx = ix + iy * nx + iz * nx * ny;
                int ker_idx = dx + dy * kx + dz * kx * ky; // C-style flat

                acc += img[img_idx] * kernel[ker_idx];
            }
        }
    }
    // Write output
    int out_idx = x + y * nx + z * nx * ny;
    out[out_idx] = acc;
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

    int nx = (int)isz[0], ny = (int)isz[1], nz = (int)isz[2];
    int kx = (int)ksz[0], ky = (int)ksz[1], kz = (int)ksz[2];

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
