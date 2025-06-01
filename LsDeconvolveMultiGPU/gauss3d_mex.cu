// gauss3d_mex.cu: Fast, VRAM-efficient, mathematically correct 3D Gaussian for Matlab GPU
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

// axis: 0=x, 1=y, 2=z
template<typename T>
__global__ void gauss3d_axis_kernel(const T* src, T* dst, const double* kernel, int klen, int nx, int ny, int nz, int axis) {
    int center = klen / 2;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int line_len;
    if (axis == 0) line_len = nx;
    else if (axis == 1) line_len = ny;
    else line_len = nz;

    for (int t = threadIdx.x; t < line_len; t += blockDim.x) {
        double val = 0.0;
        for (int k = 0; k < klen; ++k) {
            int offset = k - center;
            int ti = t + offset;
            // Clamp to valid range (replicate)
            if (axis == 0) ti = min(max(ti, 0), nx-1);
            else if (axis == 1) ti = min(max(ti, 0), ny-1);
            else ti = min(max(ti, 0), nz-1);

            size_t idx;
            if (axis == 0)      idx = z * nx * ny + y * nx + ti;
            else if (axis == 1) idx = z * nx * ny + ti * nx + x;
            else                idx = ti * nx * ny + y * nx + x;
            val += (double)src[idx] * kernel[k];
        }
        size_t oidx;
        if (axis == 0)      oidx = z * nx * ny + y * nx + t;
        else if (axis == 1) oidx = z * nx * ny + t * nx + x;
        else                oidx = t * nx * ny + y * nx + x;
        dst[oidx] = (T)val;
    }
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

        dim3 grid, block;
        if (axis == 0) { // X-axis: iterate Y,Z
            grid = dim3(nx, ny, nz); block = dim3(std::min(nx, CUDA_BLOCK_SIZE));
        } else if (axis == 1) { // Y-axis: iterate X,Z
            grid = dim3(nx, nz, 1); block = dim3(std::min(ny, CUDA_BLOCK_SIZE));
        } else { // Z-axis: iterate X,Y
            grid = dim3(nx, ny, 1); block = dim3(std::min(nz, CUDA_BLOCK_SIZE));
        }
        gauss3d_axis_kernel<T><<<grid, block>>>(src, dst, d_kernel, klen, nx, ny, nz, axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
        // Swap src/dst for next axis
        std::swap(src, dst);
    }

    // If final result is in tmp, copy back to buf
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
