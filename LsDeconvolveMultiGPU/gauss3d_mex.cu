// gauss3d_mex.cu: Minimal VRAM in-place 3D Gaussian filter for MATLAB GPU arrays
//
// - True in-place: Only a 1D temp line buffer is allocated on GPU
// - Accuracy: Double-precision accumulation for single/double input
// - Boundary: Replicate, matches MATLAB
// - VRAM: ≪ full-volume, matches imgaussfilt3 (spatial)
// - Author: ChatGPT + Keivan Moradi

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

// Gaussian kernel (host)
template<typename T>
void make_gaussian_kernel(T sigma, int ksize, T* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = (T)exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] = (T)(kernel[i] / sum);
}

// Kernel for 1D convolution of a line (in-place in line buffer)
template<typename T>
__global__ void gauss1d_line_kernel(T* line, const T* kernel, int klen, int line_len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= line_len) return;
    extern __shared__ double buf[]; // shared mem for double accumulation
    int tid = threadIdx.x;
    // Copy to double buffer
    if (idx < line_len) buf[tid] = (double)line[idx];
    __syncthreads();

    int center = klen / 2;
    double val = 0.0;
    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int ci = idx + offset;
        ci = min(max(ci, 0), line_len - 1); // replicate
        val += buf[ci] * (double)kernel[k];
    }
    __syncthreads();
    if (idx < line_len) line[idx] = (T)val;
}

// Host-side: process one axis at a time, true in-place, line by line
template<typename T>
void run_gauss3d_inplace(T* buf, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    int max_line = std::max({nx, ny, nz});
    T* h_kernel = new T[MAX_KERNEL_SIZE];
    T* d_kernel = nullptr;
    T* d_line = nullptr;
    CUDA_CHECK(cudaMalloc(&d_line, max_line * sizeof(T)));

    for (int dim = 0; dim < 3; ++dim) {
        int klen = std::min(ksize[dim], MAX_KERNEL_SIZE);
        make_gaussian_kernel(sigma[dim], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(T), cudaMemcpyHostToDevice));
        int line_len = (dim == 0) ? nx : (dim == 1) ? ny : nz;
        int n_lines = (dim == 0) ? ny * nz : (dim == 1) ? nx * nz : nx * ny;

        for (int l = 0; l < n_lines; ++l) {
            int ix = 0, iy = 0, iz = 0;
            if (dim == 0) { iy = l % ny; iz = l / ny; }
            else if (dim == 1) { ix = l % nx; iz = l / nx; }
            else { ix = l % nx; iy = l / nx; }

            // Compute start pointer of the line
            T* line_ptr;
            if (dim == 0)
                line_ptr = buf + iz * nx * ny + iy * nx;
            else if (dim == 1)
                line_ptr = buf + iz * nx * ny + ix;
            else
                line_ptr = buf + iy * nx + ix;

            // Gather the line into d_line buffer
            if (dim == 0)
                CUDA_CHECK(cudaMemcpy(d_line, line_ptr, nx * sizeof(T), cudaMemcpyDeviceToDevice));
            else if (dim == 1) {
                // Strided copy: along y
                for (int j = 0; j < ny; ++j)
                    CUDA_CHECK(cudaMemcpy(d_line + j, buf + iz * nx * ny + j * nx + ix, sizeof(T), cudaMemcpyDeviceToDevice));
            } else {
                // Strided copy: along z
                for (int k = 0; k < nz; ++k)
                    CUDA_CHECK(cudaMemcpy(d_line + k, buf + k * nx * ny + iy * nx + ix, sizeof(T), cudaMemcpyDeviceToDevice));
            }

            // Launch kernel for line
            int threads = std::min(line_len, CUDA_BLOCK_SIZE);
            int blocks = (line_len + threads - 1) / threads;
            size_t shared_mem = threads * sizeof(double);
            gauss1d_line_kernel<T><<<blocks, threads, shared_mem>>>(d_line, d_kernel, klen, line_len);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Scatter filtered line back
            if (dim == 0)
                CUDA_CHECK(cudaMemcpy(line_ptr, d_line, nx * sizeof(T), cudaMemcpyDeviceToDevice));
            else if (dim == 1) {
                for (int j = 0; j < ny; ++j)
                    CUDA_CHECK(cudaMemcpy(buf + iz * nx * ny + j * nx + ix, d_line + j, sizeof(T), cudaMemcpyDeviceToDevice));
            } else {
                for (int k = 0; k < nz; ++k)
                    CUDA_CHECK(cudaMemcpy(buf + k * nx * ny + iy * nx + ix, d_line + k, sizeof(T), cudaMemcpyDeviceToDevice));
            }
        }
        CUDA_CHECK(cudaFree(d_kernel));
    }
    CUDA_CHECK(cudaFree(d_line));
    delete[] h_kernel;
}

// MEX entry
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
