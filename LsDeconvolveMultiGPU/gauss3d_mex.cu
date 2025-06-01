// gauss3d_mex.cu: Minimal VRAM in-place 3D Gaussian filter for MATLAB GPU arrays
// Author: ChatGPT + Keivan Moradi

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#define MAX_KERNEL_SIZE 151
#define CUDA_BLOCK_SIZE 256   // May increase up to 1024 for larger lines

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

// --- 1D Gaussian kernel generation (normalized) ---
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

// --- Gather a line from 3D into d_line (parallel for Y/Z axis) ---
template<typename T>
__global__ void gather_line_kernel(const T* buf, T* d_line, int nx, int ny, int nz,
                                   int ix, int iy, int iz, int len, int dim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= len) return;
    size_t idx;
    if (dim == 1) // along Y
        idx = iz * nx * ny + i * nx + ix;
    else          // along Z
        idx = i * nx * ny + iy * nx + ix;
    d_line[i] = buf[idx];
}

// --- Scatter d_line into 3D volume (parallel for Y/Z axis) ---
template<typename T>
__global__ void scatter_line_kernel(T* buf, const T* d_line, int nx, int ny, int nz,
                                    int ix, int iy, int iz, int len, int dim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= len) return;
    size_t idx;
    if (dim == 1) // along Y
        idx = iz * nx * ny + i * nx + ix;
    else          // along Z
        idx = i * nx * ny + iy * nx + ix;
    buf[idx] = d_line[i];
}

// --- 1D convolution kernel: shared memory safe for any line length ---
template<typename T>
__global__ void gauss1d_line_kernel(T* line, const T* kernel, int klen, int line_len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= line_len) return;

    // Buffer only up to CUDA_BLOCK_SIZE
    __shared__ double buf[CUDA_BLOCK_SIZE];
    int tid = threadIdx.x;

    // Use grid-stride for long lines
    for (int base = 0; base < line_len; base += CUDA_BLOCK_SIZE) {
        int global_idx = base + tid;
        if (global_idx < line_len) {
            buf[tid] = (double)line[global_idx];
        }
        __syncthreads();

        // Convolve for this chunk, each thread for its element
        if (global_idx < line_len) {
            int center = klen / 2;
            double val = 0.0;
            for (int k = 0; k < klen; ++k) {
                int offset = k - center;
                int ci = global_idx + offset;
                // replicate boundary
                ci = min(max(ci, 0), line_len - 1);
                int local_ci = ci - base; // Local offset in shared buf
                if (local_ci >= 0 && local_ci < CUDA_BLOCK_SIZE)
                    val += buf[local_ci] * (double)kernel[k];
                else
                    val += (double)line[ci] * (double)kernel[k]; // fallback if out of chunk
            }
            line[global_idx] = (T)val;
        }
        __syncthreads();
    }
}

// --- Host-side: Axis-wise line filtering, true in-place, with safe gather/scatter ---
template<typename T>
void run_gauss3d_inplace(T* buf, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    int max_line = std::max({nx, ny, nz});
    if (max_line > 65536)
        mexErrMsgIdAndTxt("gauss3d:maxline", "Single line length exceeds supported range (65536).");
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

            // Gather line
            if (dim == 0)
                CUDA_CHECK(cudaMemcpy(d_line, buf + iz * nx * ny + iy * nx, nx * sizeof(T), cudaMemcpyDeviceToDevice));
            else {
                int threads = CUDA_BLOCK_SIZE;
                int blocks = (line_len + threads - 1) / threads;
                gather_line_kernel<T><<<blocks, threads>>>(buf, d_line, nx, ny, nz, ix, iy, iz, line_len, dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Filter the line (always full line, safe for any length)
            int threads = CUDA_BLOCK_SIZE;
            int blocks = (line_len + threads - 1) / threads;
            gauss1d_line_kernel<T><<<blocks, threads>>>(d_line, d_kernel, klen, line_len);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Scatter back
            if (dim == 0)
                CUDA_CHECK(cudaMemcpy(buf + iz * nx * ny + iy * nx, d_line, nx * sizeof(T), cudaMemcpyDeviceToDevice));
            else {
                int threads = CUDA_BLOCK_SIZE;
                int blocks = (line_len + threads - 1) / threads;
                scatter_line_kernel<T><<<blocks, threads>>>(buf, d_line, nx, ny, nz, ix, iy, iz, line_len, dim);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        CUDA_CHECK(cudaFree(d_kernel));
    }
    CUDA_CHECK(cudaFree(d_line));
    delete[] h_kernel;
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
