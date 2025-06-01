// gauss3d_mex.cu: Robust, accurate, low-VRAM separable 3D Gaussian for MATLAB GPU
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

// Generate normalized double-precision 1D Gaussian kernel (host)
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

// Each kernel launch sweeps all lines *orthogonal* to the axis, each block = one line.
// Each thread = one element in the line. Handles boundaries by replicate.
template<typename T>
__global__ void gauss1d_axis_kernel(const T* src, T* dst, const double* kernel, int klen,
                                    int nx, int ny, int nz, int axis)
{
    int line_len = (axis == 0) ? nx : (axis == 1) ? ny : nz;
    int center = klen / 2;

    // Figure out which line and position this thread is for
    int pos = threadIdx.x;
    int max_lines0 = (axis == 0) ? ny : nx;  // y for axis=0 (x-lines), x for axis=1/2
    int max_lines1 = (axis == 2) ? ny : nz;  // y for axis=2 (z-lines), z for others

    int line0 = blockIdx.x;
    int line1 = blockIdx.y;

    if (pos >= line_len) return;
    if (line0 >= max_lines0 || line1 >= max_lines1) return;

    // Compute x, y, z for this thread
    int x=0, y=0, z=0;
    if (axis == 0) {  // X lines: y=line0, z=line1, x=pos
        y = line0; z = line1; x = pos;
    } else if (axis == 1) { // Y lines: x=line0, z=line1, y=pos
        x = line0; z = line1; y = pos;
    } else { // axis == 2, Z lines: x=line0, y=line1, z=pos
        x = line0; y = line1; z = pos;
    }
    size_t idx = z * nx * ny + y * nx + x;

    // Convolve this point in the line, with boundary replicate
    double val = 0.0;
    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int pi = pos + offset;
        if (pi < 0) pi = 0;
        if (pi >= line_len) pi = line_len - 1;

        int tx=x, ty=y, tz=z;
        if (axis == 0) tx = pi;
        else if (axis == 1) ty = pi;
        else tz = pi;
        size_t sidx = tz * nx * ny + ty * nx + tx;

        val += (double)src[sidx] * kernel[k];
    }
    dst[idx] = (T)val;
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

        // For axis=0: lines = (ny, nz), len=nx
        // For axis=1: lines = (nx, nz), len=ny
        // For axis=2: lines = (nx, ny), len=nz
        int nblock0 = (axis == 0) ? ny : nx;
        int nblock1 = (axis == 2) ? ny : nz;
        int nthread = (axis == 0) ? nx : (axis == 1) ? ny : nz;

        int threads = std::min(nthread, CUDA_BLOCK_SIZE);
        dim3 grid(nblock0, nblock1, 1);
        dim3 block(threads, 1, 1);

        gauss1d_axis_kernel<T><<<grid, block>>>(src, dst, d_kernel, klen, nx, ny, nz, axis);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
        std::swap(src, dst);
    }

    // If result is in tmp, copy back to buf
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
