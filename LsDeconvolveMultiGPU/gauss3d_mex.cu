// gauss3d_mex.cu - Fast 3D Gaussian filtering with 1 extra buffer, pointer swap, CUDA + MATLAB
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

#define MAX_KERNEL_SIZE 151

// ========================
// Gaussian kernel creation
// ========================
template <typename T>
void make_gaussian_kernel(T sigma, int ksize, T* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = (T)std::exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i) kernel[i] = (T)(kernel[i] / sum);
}

// =====================
// CUDA 1D convolution
// =====================
template <typename T>
__global__ void gauss1d_kernel(
    const T* src, T* dst,
    int nx, int ny, int nz,
    const T* kernel, int klen, int axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    int line = tid / linelen;
    int pos = tid % linelen;

    // Compute x/y/z for this line/position
    int x, y, z;
    if (axis == 0) {
        y = line % ny;
        z = line / ny;
        x = pos;
    } else if (axis == 1) {
        x = line % nx;
        z = line / nx;
        y = pos;
    } else {
        x = line % nx;
        y = line / nx;
        z = pos;
    }

    int idx = x + y * nx + z * nx * ny;

    // Convolve this position along axis
    int r = klen / 2;
    double acc = 0.0;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = x, yi = y, zi = z;
        if (axis == 0) xi = min(max(x + offset, 0), nx - 1);
        if (axis == 1) yi = min(max(y + offset, 0), ny - 1);
        if (axis == 2) zi = min(max(z + offset, 0), nz - 1);
        int src_idx = xi + yi * nx + zi * nx * ny;
        acc += (double)src[src_idx] * (double)kernel[s];
    }
    dst[idx] = (T)acc;
}

// ===============================
// Host orchestration with 1 buffer
// ===============================
template <typename T>
void gauss3d_separable(
    T* input,           // in-place array (device ptr)
    T* buffer,          // extra buffer (device ptr, same size)
    int nx, int ny, int nz,
    const T sigma[3], const int ksize[3])
{
    size_t N = (size_t)nx * ny * nz;

    // Prepare Gaussian kernels
    T* h_kernel = new T[MAX_KERNEL_SIZE];
    T* d_kernel[3];
    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel[axis], ksize[axis] * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_kernel[axis], h_kernel, ksize[axis] * sizeof(T), cudaMemcpyHostToDevice));
    }
    delete[] h_kernel;

    // Swap logic: input <-> buffer
    T* src = input;
    T* dst = buffer;
    for (int axis = 0; axis < 3; ++axis) {
        int linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        int nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        int total = linelen * nline;
        int block = 256;
        int grid = (total + block - 1) / block;
        gauss1d_kernel<T><<<grid, block>>>(
            src, dst, nx, ny, nz, d_kernel[axis], ksize[axis], axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(src, dst); // Swap pointers for next axis
    }
    // Copy back if needed (final output must be in input)
    if (src != input) {
        CUDA_CHECK(cudaMemcpy(input, src, N * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    // Free device kernels
    for (int axis = 0; axis < 3; ++axis) CUDA_CHECK(cudaFree(d_kernel[axis]));
}

// ================
// MEX entry point
// ================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
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

    // Allocate one extra buffer
    void* buffer = nullptr;
    size_t N = (size_t)nx * ny * nz;
    if (cls == mxSINGLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(float)));
        float sigma[3]; for (int i = 0; i < 3; ++i) sigma[i] = (float)sigma_double[i];
        gauss3d_separable<float>((float*)ptr, (float*)buffer, nx, ny, nz, sigma, ksize);
    } else if (cls == mxDOUBLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(double)));
        double sigma[3]; for (int i = 0; i < 3; ++i) sigma[i] = sigma_double[i];
        gauss3d_separable<double>((double*)ptr, (double*)buffer, nx, ny, nz, sigma, ksize);
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }
    CUDA_CHECK(cudaFree(buffer));

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
