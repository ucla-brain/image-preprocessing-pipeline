// gauss3d_mex.cu: Correct, minimal-VRAM, accurate 3D Gaussian for MATLAB GPU
// Author: ChatGPT + Keivan Moradi

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

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

// Each CUDA kernel processes one axis, one line per block
// axis: 0=x, 1=y, 2=z
template<typename T>
__global__ void gauss1d_axis_kernel(const T* src, T* dst, const double* kernel, int klen,
                                    int nx, int ny, int nz, int axis)
{
    int line, pos;
    if (axis == 0) { // X-axis: [y,z] lines
        int y = blockIdx.x;
        int z = blockIdx.y;
        line = y + z * ny;
        pos = threadIdx.x;
        if (y >= ny || z >= nz || pos >= nx) return;
    } else if (axis == 1) { // Y-axis: [x,z] lines
        int x = blockIdx.x;
        int z = blockIdx.y;
        line = x + z * nx;
        pos = threadIdx.x;
        if (x >= nx || z >= nz || pos >= ny) return;
    } else { // Z-axis: [x,y] lines
        int x = blockIdx.x;
        int y = blockIdx.y;
        line = x + y * nx;
        pos = threadIdx.x;
        if (x >= nx || y >= ny || pos >= nz) return;
    }

    int len = (axis == 0) ? nx : (axis == 1) ? ny : nz;
    int center = klen / 2;

    // Index functions
    auto src_idx = [&](int p) -> size_t {
        if (axis == 0) return z * nx * ny + y * nx + p;
        if (axis == 1) return z * nx * ny + p * nx + x;
        return p * nx * ny + y * nx + x;
    };

    double val = 0.0;
    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int pi = pos + offset;
        // Replicate boundaries
        if (pi < 0) pi = 0;
        if (pi >= len) pi = len - 1;
        val += (double)src[src_idx(pi)] * kernel[k];
    }
    dst[src_idx(pos)] = (T)val;
}

template<typename T>
void run_gauss3d_separable(T* buf, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    size_t nvox = (size_t)nx * ny * nz;
    double h_kernel[MAX_KERNEL_SIZE];
    double* d_kernel = nullptr;
    T* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, nvox * sizeof(T)));

    T *src = buf, *dst = d_tmp;

    for (int axis = 0; axis < 3; ++axis) {
        int klen = std::min(ksize[axis], MAX_KERNEL_SIZE);
        make_gaussian_kernel(sigma[axis], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(double), cudaMemcpyHostToDevice));

        int nblock0, nblock1, nthread;
        if (axis == 0) { // X-axis: [y,z] lines
            nblock0 = ny; nblock1 = nz; nthread = nx;
        } else if (axis == 1) { // Y-axis: [x,z] lines
            nblock0 = nx; nblock1 = nz; nthread = ny;
        } else { // Z-axis: [x,y] lines
            nblock0 = nx; nblock1 = ny; nthread = nz;
        }
        int threads = std::min(nthread, CUDA_BLOCK_SIZE);
        int blocks = (nthread + threads - 1) / threads;
        dim3 grid(nblock0, nblock1, 1);
        dim3 block(threads, 1, 1);

        gauss1d_axis_kernel<T><<<grid, block>>>(src, dst, d_kernel, klen, nx, ny, nz, axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
        std::swap(src, dst);
    }

    // If last swap left the result in tmp, copy back to buf
    if (src != buf)
        CUDA_CHECK(cudaMemcpy(buf, src, nvox * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_tmp));
}

// --- MEX entry point ---
extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size])");

    const mxGPUArray* img_gpu_const = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray* img_gpu = const_cast<mxGPUArray*>(img_gpu_const);
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
        run_gauss3d_separable<float>((float*)ptr, nx, ny, nz, sigma, ksize);
    } else if (cls == mxDOUBLE_CLASS) {
        double sigma[3];
        for (int i = 0; i < 3; ++i) sigma[i] = sigma_double[i];
        run_gauss3d_separable<double>((double*)ptr, nx, ny, nz, sigma, ksize);
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
