// gauss3d_mex.cu - Correct, high-precision, fast 3D Gaussian (1 temp buffer)
// Author: ChatGPT + Keivan Moradi

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#define MAX_KERNEL_SIZE 151
#define CUDA_BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

void make_gaussian_kernel(double sigma, int ksize, double* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] /= sum;
}

// CUDA kernel: Apply 1D Gaussian to every line along given axis
template<typename T>
__global__ void gauss1d_kernel(const T* src, T* dst, const double* kernel, int klen,
                               int nx, int ny, int nz, int axis)
{
    // Find which line (out of all possible lines along 'axis') this block processes:
    int n_lines[3] = { ny*nz, nx*nz, nx*ny };
    int line_idx = blockIdx.x;
    int pos = threadIdx.x;
    int line_len = (axis==0) ? nx : (axis==1) ? ny : nz;

    if (line_idx >= n_lines[axis] || pos >= line_len) return;

    // For each axis, decode which indices this line maps to:
    int x=0, y=0, z=0;
    if (axis == 0) { // X lines
        y = line_idx % ny;
        z = line_idx / ny;
        x = pos;
    } else if (axis == 1) { // Y lines
        x = line_idx % nx;
        z = line_idx / nx;
        y = pos;
    } else { // Z lines
        x = line_idx % nx;
        y = line_idx / nx;
        z = pos;
    }
    size_t idx = z*nx*ny + y*nx + x;

    // Convolve the line with replicate boundary:
    int center = klen / 2;
    double val = 0.0;
    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int pi = pos + offset;
        if (pi < 0) pi = 0;
        if (pi >= line_len) pi = line_len - 1;
        int tx=x, ty=y, tz=z;
        if (axis == 0) tx=pi;
        else if (axis == 1) ty=pi;
        else tz=pi;
        size_t sidx = tz*nx*ny + ty*nx + tx;
        val += (double)src[sidx] * kernel[k];
    }
    dst[idx] = (T)val;
}

template<typename T>
void run_gauss3d_separable(T* buf, int nx, int ny, int nz, const double sigma[3], const int ksize[3]) {
    size_t nvox = (size_t)nx*ny*nz;
    double h_kernel[MAX_KERNEL_SIZE];
    double* d_kernel = nullptr;
    T* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, nvox * sizeof(T)));

    T* src = buf;
    T* dst = d_tmp;

    for (int axis = 0; axis < 3; ++axis) {
        int klen = std::min(ksize[axis], MAX_KERNEL_SIZE);
        make_gaussian_kernel(sigma[axis], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(double), cudaMemcpyHostToDevice));

        int n_lines = (axis==0) ? ny*nz : (axis==1) ? nx*nz : nx*ny;
        int line_len = (axis==0) ? nx : (axis==1) ? ny : nz;
        int threads = std::min(line_len, CUDA_BLOCK_SIZE);
        int blocks = n_lines;
        gauss1d_kernel<T><<<blocks, threads>>>(src, dst, d_kernel, klen, nx, ny, nz, axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
        std::swap(src, dst);
    }
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
        run_gauss3d_separable<float>((float*)ptr, nx, ny, nz, sigma_double, ksize);
    } else if (cls == mxDOUBLE_CLASS) {
        run_gauss3d_separable<double>((double*)ptr, nx, ny, nz, sigma_double, ksize);
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
