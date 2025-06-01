// gauss3d_mex.cu: Robust in-place 3D Gaussian filter for MATLAB GPU arrays
//
// Features:
//   - Separable convolution (x, y, z axes) using custom kernel
//   - Works for single or double precision (gpuArray)
//   - Double-precision accumulation for accuracy (even on float input)
//   - Robust for very large volumes (grid-stride loop in kernel)
//   - Thorough input and kernel checks
//   - Replicate boundary handling (matches MATLAB default)
//   - Descriptive comments and error handling for maintenance
//
// Author: ChatGPT & Keivan Moradi
// Date: 2024

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <limits>

#define MAX_KERNEL_SIZE 151  // Change as needed, must match your largest expected filter

#define CUDA_BLOCK_SIZE 256  // Safe default for most GPUs

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)


// ---- Kernel generator: normalized 1D Gaussian ----
template<typename T>
void make_gaussian_kernel(T sigma, int ksize, T* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = (T)exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + r];
    }
    // Normalize so sum(kernel)==1
    for (int i = 0; i < ksize; ++i)
        kernel[i] = (T)(kernel[i] / sum);
}

// ---- Check if kernel size is too small for sigma ----
void warn_kernel_size(const double* sigma, const int* ksize, int do_warn) {
    for (int i = 0; i < 3; ++i) {
        int min_ksize = 2 * static_cast<int>(ceil(3.0 * sigma[i])) + 1;
        if (ksize[i] < min_ksize && do_warn) {
            mexWarnMsgIdAndTxt("gauss3d:kernelSizeTooSmall",
                "Kernel size for axis %d (%d) is too small for sigma=%.3f (recommended at least %d). Results may be inaccurate.\n"
                "To disable this warning, call gauss3d_mex(..., ..., ..., true) to disable warnings.",
                i + 1, ksize[i], sigma[i], min_ksize);
        }
        if (ksize[i] > MAX_KERNEL_SIZE) {
            mexWarnMsgIdAndTxt("gauss3d:kernelSizeTooLarge",
                "Requested kernel size for axis %d (%d) exceeds MAX_KERNEL_SIZE (%d). Limiting to %d.",
                i + 1, ksize[i], MAX_KERNEL_SIZE, MAX_KERNEL_SIZE);
        }
    }
}

// ---- CUDA KERNEL: Each block handles a line; grid-stride for robustness ----
template<typename T>
__global__ void gauss1d_lines_kernel(
    const T* __restrict__ data_in, // Input buffer (const!)
    T* data_out,                   // Output buffer
    const T* __restrict__ kernel, int klen,
    int nx, int ny, int nz,
    int dim, int line_len, int n_lines)
{
    int line_idx = blockIdx.x;

    // Compute fixed coordinates for this line in 3D space
    int ix0 = 0, iy0 = 0, iz0 = 0;
    if (dim == 0) { iy0 = line_idx % ny; iz0 = line_idx / ny; }
    else if (dim == 1) { ix0 = line_idx % nx; iz0 = line_idx / nx; }
    else { ix0 = line_idx % nx; iy0 = line_idx / nx; }

    int center = klen / 2;
    // Handle lines longer than block size with grid-stride loop
    for (int idx_in_line = threadIdx.x; idx_in_line < line_len; idx_in_line += blockDim.x) {
        // Output position in full 3D volume
        int ix = ix0, iy = iy0, iz = iz0;
        if (dim == 0) ix = idx_in_line;
        else if (dim == 1) iy = idx_in_line;
        else iz = idx_in_line;

        // Robust boundary replicate logic
        double val = 0.0;
        for (int k = 0; k < klen; ++k) {
            int offset = k - center;
            int ci = idx_in_line + offset;
            ci = min(max(ci, 0), line_len - 1); // replicate boundary

            int cx = ix0, cy = iy0, cz = iz0;
            if (dim == 0) cx = ci;
            else if (dim == 1) cy = ci;
            else cz = ci;
            int in_idx = cz * nx * ny + cy * nx + cx;

            val += static_cast<double>(data_in[in_idx]) * static_cast<double>(kernel[k]);
        }
        int out_idx = iz * nx * ny + iy * nx + ix;
        data_out[out_idx] = static_cast<T>(val);
    }
}

// ---- Separable 3D Gaussian filtering with robustness ----
template<typename T>
void run_gauss3d(T* buf, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    size_t nvox = static_cast<size_t>(nx) * ny * nz;
    T* h_kernel = new T[MAX_KERNEL_SIZE];
    T* d_kernel = nullptr;

    // Allocate a temp buffer on the GPU for out-of-place computation
    T* buf_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&buf_tmp, nvox * sizeof(T)));

    T* src = buf;
    T* dst = buf_tmp;

    for (int dim = 0; dim < 3; ++dim) {
        int klen = std::min(ksize[dim], MAX_KERNEL_SIZE);  // Limit kernel size for safety
        make_gaussian_kernel(sigma[dim], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(T), cudaMemcpyHostToDevice));

        size_t n_lines, line_len;
        if (dim == 0) { n_lines = (size_t)ny * nz; line_len = nx; }
        else if (dim == 1) { n_lines = (size_t)nx * nz; line_len = ny; }
        else { n_lines = (size_t)nx * ny; line_len = nz; }

        // CUDA grid-dim limit: 65535 x 65535
        const size_t CUDA_GRID_MAX = 65535ULL * 65535ULL;
        if (n_lines > CUDA_GRID_MAX)
            mexErrMsgIdAndTxt("gauss3d:gridSize", "Too many lines for CUDA kernel launch (max: %zu).", CUDA_GRID_MAX);


        int block_size = (line_len < CUDA_BLOCK_SIZE) ? line_len : CUDA_BLOCK_SIZE;
        dim3 block(block_size);
        dim3 grid(n_lines);

        gauss1d_lines_kernel<T><<<grid, block>>>(
            src, dst, d_kernel, klen, nx, ny, nz, dim, line_len, n_lines
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));

        // Swap src and dst for next pass
        T* tmp = src;
        src = dst;
        dst = tmp;
    }

    // After 3 passes (odd), src points to buf_tmp, copy back if needed
    if (src != buf) {
        CUDA_CHECK(cudaMemcpy(buf, buf_tmp, nvox * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(buf_tmp));
    delete[] h_kernel;
}

// ---- Main MATLAB MEX entry point ----
extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    // ---- Input validation ----
    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size, disable_warning])");

    const mxGPUArray* img_gpu_const = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray* img_gpu = const_cast<mxGPUArray*>(img_gpu_const); // No copy, direct access
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3) mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D. Got %dD array.", nd);
    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];
    if (nx < 1 || ny < 1 || nz < 1)
        mexErrMsgIdAndTxt("gauss3d:empty", "Input array has a zero-sized dimension.");

    // ---- Parse sigma ----
    double sigma_double[3];
    if (mxIsScalar(prhs[1])) {
        double v = mxGetScalar(prhs[1]);
        if (!(v > 0))
            mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma must be > 0.");
        sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
    } else if (mxGetNumberOfElements(prhs[1]) == 3) {
        double* s = mxGetPr(prhs[1]);
        for (int i = 0; i < 3; ++i) {
            if (!(s[i] > 0))
                mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma(%d) must be > 0.", i + 1);
            sigma_double[i] = s[i];
        }
    } else {
        mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma must be scalar or 3-vector.");
    }

    // ---- Parse kernel_size (optional) ----
    int ksize[3];
    if (nrhs >= 3 && !mxIsLogicalScalar(prhs[2])) {
        if (mxIsScalar(prhs[2])) {
            int k = (int)mxGetScalar(prhs[2]);
            if (k < 1) mexErrMsgIdAndTxt("gauss3d:kernel", "Kernel size must be positive.");
            ksize[0] = ksize[1] = ksize[2] = k;
        } else if (mxGetNumberOfElements(prhs[2]) == 3) {
            double* ks = mxGetPr(prhs[2]);
            for (int i = 0; i < 3; ++i) {
                if (!(ks[i] >= 1)) mexErrMsgIdAndTxt("gauss3d:kernel", "Kernel size(%d) must be positive.", i + 1);
                ksize[i] = (int)ks[i];
            }
        } else {
            mexErrMsgIdAndTxt("gauss3d:kernel", "Kernel size must be scalar or 3-vector.");
        }
    } else {
        for (int i = 0; i < 3; ++i)
            ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
    }

    // ---- Parse warning preference (4th argument, optional) ----
    int do_warn = 1;
    if (nrhs >= 4 && mxIsLogicalScalar(prhs[3])) {
        do_warn = !mxIsLogicalScalarTrue(prhs[3]);
    } else {
        const char* warn_env = std::getenv("GAUSS3D_WARN_KSIZE");
        if (warn_env && warn_env[0] == '0') do_warn = 0;
    }
    warn_kernel_size(sigma_double, ksize, do_warn);

    // ---- Run filtering: class dispatch ----
    mxClassID cls = mxGPUGetClassID(img_gpu);
    void* ptr = mxGPUGetData(img_gpu);

    if (cls == mxSINGLE_CLASS) {
        float sigma[3];
        for (int i = 0; i < 3; ++i) sigma[i] = (float)sigma_double[i];
        run_gauss3d<float>((float*)ptr, nx, ny, nz, sigma, ksize);
    } else if (cls == mxDOUBLE_CLASS) {
        double sigma[3];
        for (int i = 0; i < 3; ++i) sigma[i] = sigma_double[i];
        run_gauss3d<double>((double*)ptr, nx, ny, nz, sigma, ksize);
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double precision gpuArray.");
    }

    // ---- Return modified input as output ----
    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
