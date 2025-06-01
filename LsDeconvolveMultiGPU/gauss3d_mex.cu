// gauss3d_mex.cu: Fast, VRAM-minimal, Matlab-accurate in-place 3D Gaussian filter for MATLAB GPU arrays
// Author: ChatGPT + Keivan Moradi

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#define MAX_KERNEL_SIZE 151
#define CUDA_BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

template<typename T>
void make_gaussian_kernel(T sigma, int ksize, double* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] /= sum;
}

// X-axis: fully coalesced, 1 block per row, 1 thread per column
template<typename T>
__global__ void gauss3d_axis0_kernel(T* vol, const double* kernel, int klen, int nx, int ny, int nz) {
    int row = blockIdx.x;
    int plane = blockIdx.y;
    int y = row, z = plane;
    int center = klen / 2;

    if (y >= ny || z >= nz) return;

    T* line = vol + z * nx * ny + y * nx;

    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        double val = 0.0;
        for (int k = 0; k < klen; ++k) {
            int offset = k - center;
            int xi = x + offset;
            xi = min(max(xi, 0), nx - 1); // replicate
            val += (double)line[xi] * kernel[k];
        }
        __syncthreads(); // all threads update after all reads (not strictly needed here)
        line[x] = (T)val;
    }
}

// Y-axis: each thread handles one line (not coalesced, but batched)
template<typename T>
__global__ void gauss3d_axis1_kernel(T* vol, const double* kernel, int klen, int nx, int ny, int nz) {
    int col = blockIdx.x;
    int plane = blockIdx.y;
    int x = col, z = plane;
    int center = klen / 2;

    if (x >= nx || z >= nz) return;

    for (int y = threadIdx.x; y < ny; y += blockDim.x) {
        double val = 0.0;
        for (int k = 0; k < klen; ++k) {
            int offset = k - center;
            int yi = y + offset;
            yi = min(max(yi, 0), ny - 1);
            size_t idx = z * nx * ny + yi * nx + x;
            val += (double)vol[idx] * kernel[k];
        }
        size_t idx_out = z * nx * ny + y * nx + x;
        vol[idx_out] = (T)val;
    }
}

// Z-axis: each thread handles one line
template<typename T>
__global__ void gauss3d_axis2_kernel(T* vol, const double* kernel, int klen, int nx, int ny, int nz) {
    int col = blockIdx.x;
    int row = blockIdx.y;
    int x = col, y = row;
    int center = klen / 2;

    if (x >= nx || y >= ny) return;

    for (int z = threadIdx.x; z < nz; z += blockDim.x) {
        double val = 0.0;
        for (int k = 0; k < klen; ++k) {
            int offset = k - center;
            int zi = z + offset;
            zi = min(max(zi, 0), nz - 1);
            size_t idx = zi * nx * ny + y * nx + x;
            val += (double)vol[idx] * kernel[k];
        }
        size_t idx_out = z * nx * ny + y * nx + x;
        vol[idx_out] = (T)val;
    }
}

template<typename T>
void run_gauss3d_inplace(T* buf, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    double h_kernel[MAX_KERNEL_SIZE];
    double* d_kernel = nullptr;

    for (int dim = 0; dim < 3; ++dim) {
        int klen = std::min(ksize[dim], MAX_KERNEL_SIZE);
        make_gaussian_kernel(sigma[dim], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(double), cudaMemcpyHostToDevice));

        dim3 grid, block;
        if (dim == 0) { // X-axis
            grid = dim3(ny, nz);
            block = dim3(std::min(nx, CUDA_BLOCK_SIZE));
            gauss3d_axis0_kernel<T><<<grid, block>>>(buf, d_kernel, klen, nx, ny, nz);
        } else if (dim == 1) { // Y-axis
            grid = dim3(nx, nz);
            block = dim3(std::min(ny, CUDA_BLOCK_SIZE));
            gauss3d_axis1_kernel<T><<<grid, block>>>(buf, d_kernel, klen, nx, ny, nz);
        } else { // Z-axis
            grid = dim3(nx, ny);
            block = dim3(std::min(nz, CUDA_BLOCK_SIZE));
            gauss3d_axis2_kernel<T><<<grid, block>>>(buf, d_kernel, klen, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
    }
}

// --- MEX entry point ---
extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size])");

    const mxGPUArray* img_gpu_const = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray* img_gpu = const_cast<mxGPUArray*>(img_gpu_const); // No copy
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3) mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D.");
    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];

    double sigma_double[3];
    if (mxIsScalar(prhs[1])) {
        double v = mxGetScalar(prhs[1]);
        sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
    } else if (mxGetNumberOfElements(prhs[1]) == 3) {
        double* s = mxGetPr(prhs[1]);
        for (int i = 0; i < 3; ++i) sigma_double[i] = s[i];
    } else {
        mexErrMsgIdAndTxt("gauss3d:sigma", "sigma must be scalar or 3-vector");
    }

    int ksize[3];
    if (nrhs >= 3 && !mxIsLogicalScalar(prhs[2])) {
        if (mxIsScalar(prhs[2])) {
            int k = (int)mxGetScalar(prhs[2]);
            ksize[0] = ksize[1] = ksize[2] = k;
        } else if (mxGetNumberOfElements(prhs[2]) == 3) {
            double* ks = mxGetPr(prhs[2]);
            for (int i = 0; i < 3; ++i) ksize[i] = (int)ks[i];
        } else {
            mexErrMsgIdAndTxt("gauss3d:kernel", "kernel_size must be scalar or 3-vector");
        }
    } else {
        for (int i = 0; i < 3; ++i)
            ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
    }

    mxClassID cls = mxGPUGetClassID(img_gpu);
    void* ptr = mxGPUGetData(img_gpu);

    if (cls == mxSINGLE_CLASS) {
        float sigma[3];
        for (int i = 0; i < 3; ++i) sigma[i] = (float)sigma_double[i];
        run_gauss3d_inplace<float>((float*)ptr, nx, ny, nz, sigma, ksize);
    } else if (cls == mxDOUBLE_CLASS) {
        double sigma[3];
        for (int i = 0; i < 3; ++i) sigma[i] = sigma_double[i];
        run_gauss3d_inplace<double>((double*)ptr, nx, ny, nz, sigma, ksize);
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
