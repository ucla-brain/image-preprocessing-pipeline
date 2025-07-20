/*==============================================================================
  fused_gauss3d_gpu.cu

  High-Performance 3D Gaussian Filtering (Single-Precision, Fused, GPU-Accelerated)
  MATLAB MEX implementation with dynamic, hardware-adaptive shared memory tiling,
  register blocking, and cache-aware block reordering.

  -------------------------------------------------------------------------------
  Features & Optimizations:
    - **Fused** single-pass 3D Gaussian convolution using shared memory tiling
    - **Column-major** input/output (MATLAB/Numpy compatibility)
    - **Dynamic tile/block size** selection based on GPU properties and input size
    - **Register blocking** (REG_X): each thread computes multiple outputs per pass
    - **Y-major grid/block ordering** for improved L2/L1 cache locality (empirically best for large volumes)
    - **Shared memory bank conflict avoidance** (+1 stride on X axis)
    - **Thread-strided persistent loads** for optimal occupancy (avoids redundant global reads)
    - **fmaf (Fused Multiply-Add)** in innermost loop for best precision & performance
    - **Automatic kernel size calculation** based on sigma (user-supplied kernel sizes are not supported for optimal path)
    - **Auto-handle device shared memory and thread block limits**
    - **Clamp boundary conditions** (extends border)
    - **Comprehensive MATLAB test harness** for accuracy and speedup (see `gauss3d_gpu_test.m`)

  -------------------------------------------------------------------------------
  Authors:
    Keivan Moradi (2025)
    ChatGPT (OpenAI, code co-design and optimization)
  License: MIT or academic use

  -------------------------------------------------------------------------------
  Example Usage (MATLAB):
      build_mex; gauss3d_gpu_test
      % or:
      y = gauss3d_gpu(gpuArray(single(x)), [sx sy sz]);

  -------------------------------------------------------------------------------
  For detailed benchmark and test setup, see:
      gauss3d_gpu_test.m
==============================================================================*/


#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <tuple>

#define MAX_KERNEL_RADIUS 25   // Support up to 51x51x51 kernel

__constant__ float gaussX[2*MAX_KERNEL_RADIUS+1];
__constant__ float gaussY[2*MAX_KERNEL_RADIUS+1];
__constant__ float gaussZ[2*MAX_KERNEL_RADIUS+1];

__device__ __forceinline__ int clamp(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

// --- Fused 3D Gaussian convolution kernel, Y-major grid order ---
__global__
void fused_gauss3d_shared_kernel(const float* __restrict__ input, float* __restrict__ output,
    int nx, int ny, int nz,
    int kx, int ky, int kz,
    int rx, int ry, int rz,
    int tX, int tY, int tZ,
    int REG_X)
{
    // --- Y-major grid: blockIdx.x = Y, blockIdx.y = Z, blockIdx.z = X
    int by = blockIdx.x, bz = blockIdx.y, bx = blockIdx.z;
    int ty = threadIdx.x, tz = threadIdx.y, tx = threadIdx.z;

    // Each thread computes REG_X outputs along X
    int x0 = bx * tX + tx * REG_X;
    int y  = by * tY + ty;
    int z  = bz * tZ + tz;

    // Shared memory tile with halo (+1 for bank conflict avoidance)
    int sNx = tX + 2*rx + 1;
    int sNy = tY + 2*ry;
    int sNz = tZ + 2*rz;
    extern __shared__ float tile[];

    // Thread-strided persistent loading (flattened)
    int tileElems = sNx * sNy * sNz;
    int blockThreads = (tX/REG_X) * tY * tZ; // updated for REG_X
    int threadIdx3D = tz * (tX/REG_X * tY) + ty * (tX/REG_X) + tx;

    for (int idx = threadIdx3D; idx < tileElems; idx += blockThreads) {
        int dz = idx / (sNx * sNy);
        int dy = (idx / sNx) % sNy;
        int dx = idx % sNx;
        int gx = clamp(bx * tX + dx - rx, 0, nx-1);
        int gy = clamp(by * tY + dy - ry, 0, ny-1);
        int gz = clamp(bz * tZ + dz - rz, 0, nz-1);
        tile[(dx) + (dy)*sNx + (dz)*sNx*sNy] = input[gx + gy*nx + gz*nx*ny];
    }
    __syncthreads();

    // Register-blocked output: each thread computes REG_X outputs along X
    int sy = ty + ry;
    int sz = tz + rz;
    int sx0 = tx * REG_X + rx;
    #pragma unroll
    for (int reg = 0; reg < REG_X; ++reg) {
        int x = x0 + reg;
        int sx = sx0 + reg;

        if (x < nx && y < ny && z < nz) {
            float acc = 0.0f;
            for (int dz = -rz; dz <= rz; ++dz) {
                int szp = sz + dz;
                float kzv = gaussZ[dz + rz];
                for (int dy = -ry; dy <= ry; ++dy) {
                    int syp = sy + dy;
                    float kyv = gaussY[dy + ry];
                    #pragma unroll
                    for (int dx = -rx; dx <= rx; ++dx) {
                        int sxp = sx + dx;
                        float kxv = gaussX[dx + rx];
                        int sIdx = sxp + syp*sNx + szp*sNx*sNy;
                        acc = fmaf(tile[sIdx], kxv * kyv * kzv, acc);
                    }
                }
            }
            int outIdx = x + y*nx + z*nx*ny;
            output[outIdx] = acc;
        }
    }
}

// Helper: Gaussian kernel
void make_gaussian_kernel(float sigma, int ksize, float* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] = static_cast<float>(kernel[i] / sum);
}

// Helper: Robust kernel size calculation
std::array<int,3> returnKernelSize(const mxArray* prhs2, const double sigma_double[3]) {
    std::array<int,3> ksize;
    for (int i = 0; i < 3; ++i)
        ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
    for (int i = 0; i < 3; ++i) {
        if (ksize[i] < 3) ksize[i] = 3;
        if (ksize[i] % 2 == 0) ksize[i] += 1;
    }
    return ksize;
}

std::tuple<int, int, int, size_t, int>
chooseTileAndSharedMem(int rx, int ry, int rz, int nx, int ny, int nz, const char* errMsgPrefix)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t maxSMem = prop.sharedMemPerBlock;
    const int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    int REG_X;
    if (nx >= 128)      REG_X = 4;
    else if (nx >= 64)  REG_X = 2;
    else                REG_X = 1;

    int tX = 8 * REG_X, tY = 8, tZ = 4;
    int maxRad = std::max({rx, ry, rz});
    if (nx >= 256)      tX = 16 * REG_X;
    if (ny >= 256)      tY = 16;
    if (nz >= 128)      tZ = 8;

    if (maxRad > 15) { tX = 4 * REG_X; tY = 4; tZ = 2; }

    int minTile = 1;

    while ((tX/REG_X * tY * tZ) > maxThreadsPerBlock) {
        if (tX >= tY && tX >= tZ && tX > minTile*REG_X) tX /= 2;
        else if (tY >= tX && tY >= tZ && tY > minTile) tY /= 2;
        else if (tZ > minTile) tZ /= 2;
    }

    int sNx = tX + 2*rx + 1; // +1 for bank conflict avoidance
    int sNy = tY + 2*ry;
    int sNz = tZ + 2*rz;
    size_t sharedMemSize = (size_t)sNx * sNy * sNz * sizeof(float);

    while (sharedMemSize > maxSMem && (tX > minTile*REG_X || tY > minTile || tZ > minTile)) {
        if (tX >= tY && tX >= tZ && tX > minTile*REG_X) tX /= 2;
        else if (tY >= tX && tY >= tZ && tY > minTile) tY /= 2;
        else if (tZ > minTile) tZ /= 2;
        sNx = tX + 2*rx + 1;
        sNy = tY + 2*ry;
        sNz = tZ + 2*rz;
        sharedMemSize = (size_t)sNx * sNy * sNz * sizeof(float);
        if ((tX/REG_X * tY * tZ) > maxThreadsPerBlock) {
            if (tX >= tY && tX >= tZ && tX > minTile*REG_X) tX /= 2;
            else if (tY >= tX && tY >= tZ && tY > minTile) tY /= 2;
            else if (tZ > minTile) tZ /= 2;
        }
    }

    if (sharedMemSize > maxSMem || (tX/REG_X * tY * tZ) > maxThreadsPerBlock) {
        char errId[128];
        snprintf(errId, sizeof(errId), "%s:sharedmem", errMsgPrefix);
        mexErrMsgIdAndTxt(
            errId,
            "Requested kernel/tile size needs too much shared memory (%zu bytes, %d threads, REG_X=%d), "
            "device limit is %zu bytes, %d threads. Try a smaller kernel.",
            sharedMemSize, tX/REG_X * tY * tZ, REG_X, maxSMem, maxThreadsPerBlock
        );
    }

    return std::make_tuple(tX, tY, tZ, sharedMemSize, REG_X);
}

extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();

    if (nrhs < 2)
        mexErrMsgIdAndTxt("fused_gauss3d_gpu:", "Usage: fused_gauss3d_gpu(x, sigma)");

    mxGPUArray* img_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3)
        mexErrMsgIdAndTxt("fused_gauss3d_gpu:", "Input must be 3D.");

    int nx = sz[0], ny = sz[1], nz = sz[2];
    size_t N = static_cast<size_t>(nx) * ny * nz;
    mxClassID cls = mxGPUGetClassID(img_gpu);
    void* ptr = mxGPUGetData(img_gpu);
    if (cls != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("fused_gauss3d_gpu:", "Input must be single-precision gpuArray");

    // Read sigma argument
    double sigma_double[3];
    if (mxIsScalar(prhs[1])) {
        double v = mxGetScalar(prhs[1]);
        sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
    } else if (mxGetNumberOfElements(prhs[1]) == 3) {
        double* s = mxGetPr(prhs[1]);
        for (int i = 0; i < 3; ++i) sigma_double[i] = s[i];
    } else {
        mexErrMsgIdAndTxt("fused_gauss3d_gpu:", "sigma must be scalar or 3-vector");
    }

    // Robust kernel size
    std::array<int,3> ksize = returnKernelSize(nullptr, sigma_double);
    int rx = ksize[0]/2, ry = ksize[1]/2, rz = ksize[2]/2;

    // Prepare Gaussian kernels
    float h_kx[51], h_ky[51], h_kz[51];
    make_gaussian_kernel((float)sigma_double[0], ksize[0], h_kx);
    make_gaussian_kernel((float)sigma_double[1], ksize[1], h_ky);
    make_gaussian_kernel((float)sigma_double[2], ksize[2], h_kz);

    cudaMemcpyToSymbol(gaussX, h_kx, ksize[0]*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gaussY, h_ky, ksize[1]*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gaussZ, h_kz, ksize[2]*sizeof(float), 0, cudaMemcpyHostToDevice);

    // Allocate output buffer
    float* d_output;
    cudaError_t err = cudaMalloc(&d_output, N*sizeof(float));
    if (err != cudaSuccess)
        mexErrMsgIdAndTxt("fused_gauss3d_gpu:cudaMalloc", "Failed to allocate output buffer (%s)", cudaGetErrorString(err));

    // === Dynamic tile size selection ===
    int tX, tY, tZ, REG_X;
    size_t sharedMemSize;
    std::tie(tX, tY, tZ, sharedMemSize, REG_X) = chooseTileAndSharedMem(rx, ry, rz, nx, ny, nz, "fused_gauss3d_gpu");

    // --------- Y-major launch: blockIdx.x = Y, blockIdx.y = Z, blockIdx.z = X
    dim3 blockDim(tY, tZ, tX/REG_X);
    dim3 gridDim((ny + tY - 1)/tY, (nz + tZ - 1)/tZ, (nx + tX - 1)/tX);

    fused_gauss3d_shared_kernel<<<gridDim, blockDim, sharedMemSize>>>(
        (const float*)ptr, d_output,
        nx, ny, nz,
        ksize[0], ksize[1], ksize[2],
        rx, ry, rz,
        tX, tY, tZ,
        REG_X
    );
    cudaDeviceSynchronize();

    // Return result as a new gpuArray
    mxGPUArray* out_gpu = mxGPUCreateGPUArray(nd, sz, mxSINGLE_CLASS,
        mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    void* out_ptr = mxGPUGetData(out_gpu);
    cudaMemcpy(out_ptr, d_output, N*sizeof(float), cudaMemcpyDeviceToDevice);

    plhs[0] = mxGPUCreateMxArrayOnGPU(out_gpu);

    cudaFree(d_output);
    mxGPUDestroyGPUArray(img_gpu);
    mxGPUDestroyGPUArray(out_gpu);
}
