// gauss3d_mex.cu: In-place 3D Gaussian filter for MATLAB, with warnings and kernel normalization

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>

#define MAX_KERNEL_SIZE 151

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error: %s", cudaGetErrorString(err)); \
} while(0)

// --------- KERNEL GENERATION & NORMALIZATION (float/double) ---------
template<typename T>
void make_gaussian_kernel(T sigma, int ksize, T* kernel) {
    int r = ksize / 2;
    T sum = 0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = exp(-0.5 * (i*i) / (sigma*sigma));
        sum += kernel[i + r];
    }
    // Normalize kernel so sum(kernel)==1
    for (int i = 0; i < ksize; ++i)
        kernel[i] /= sum;
}

// --------- CUDA KERNEL (accumulate in T) ---------
template <typename T>
__global__ void gauss1d_kernel(
    T* src, T* dst, int nx, int ny, int nz,
    const T* kernel, int klen, int dim
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int size[3] = {nx, ny, nz};
    int center = klen / 2;

    int idx = iz * nx * ny + iy * nx + ix;
    T val = 0;

    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int ci = (dim == 0) ? ix + offset : (dim == 1) ? iy + offset : iz + offset;
        ci = min(max(ci, 0), size[dim] - 1); // replicate boundary
        int cidx;
        if (dim == 0)
            cidx = iz * nx * ny + iy * nx + ci;
        else if (dim == 1)
            cidx = iz * nx * ny + ci * nx + ix;
        else
            cidx = ci * nx * ny + iy * nx + ix;
        val += src[cidx] * kernel[k];
    }
    dst[idx] = val;
}

// --------- WARN IF KERNEL SIZE TOO SMALL ---------
void warn_kernel_size(const double* sigma, const int* ksize, int do_warn) {
    for (int i = 0; i < 3; ++i) {
        int min_ksize = 2 * static_cast<int>(ceil(3.0 * sigma[i])) + 1;
        if (ksize[i] < min_ksize && do_warn) {
            mexWarnMsgIdAndTxt("gauss3d:kernelSizeTooSmall",
                "Kernel size for axis %d (%d) is too small for sigma=%.3f (recommended at least %d). Results may be inaccurate.\n"
                "To disable this warning, call gauss3d_mex(..., ..., ..., true) to disable warnings.",
                i+1, ksize[i], sigma[i], min_ksize);
        }
    }
}

// --------- SEPARABLE GAUSS 3D (in-place) ---------
template<typename T>
void run_gauss3d_inplace(T* bufA, int nx, int ny, int nz, const T sigma[3], const int ksize[3]) {
    T* h_kernel = new T[MAX_KERNEL_SIZE];
    T* d_kernel;
    int klen;

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    // Allocate temp buffer for swap
    T* bufB;
    CUDA_CHECK(cudaMalloc(&bufB, nx*ny*nz*sizeof(T)));

    T *src = bufA;
    T *dst = bufB;
    for (int dim = 0; dim < 3; ++dim) {
        klen = ksize[dim];
        if (klen > MAX_KERNEL_SIZE)
            mexErrMsgIdAndTxt("gauss3d:kernel", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
        make_gaussian_kernel(sigma[dim], klen, h_kernel);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(T), cudaMemcpyHostToDevice));

        gauss1d_kernel<T><<<grid, block>>>(src, dst, nx, ny, nz, d_kernel, klen, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));

        // Swap
        T* tmp = src; src = dst; dst = tmp;
    }

    // After 3 passes (odd), result is in bufB, copy back to bufA
    if (src != bufA) {
        CUDA_CHECK(cudaMemcpy(bufA, bufB, nx*ny*nz*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaFree(bufB));
    delete[] h_kernel;
}

// --------- MEX ENTRY ---------
extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size, disable_warning])");

    // Parse input array (in-place)
    const mxGPUArray* img_gpu_const = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray* img_gpu = const_cast<mxGPUArray*>(img_gpu_const);
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3) mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D");
    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];

    // --- Parse sigma ---
    double sigma_double[3];
    if (mxIsScalar(prhs[1])) {
        double v = mxGetScalar(prhs[1]);
        sigma_double[0] = sigma_double[1] = sigma_double[2] = v;
    } else if (mxGetNumberOfElements(prhs[1]) == 3) {
        double* s = mxGetPr(prhs[1]);
        for(int i=0; i<3; ++i) sigma_double[i] = s[i];
    } else {
        mexErrMsgIdAndTxt("gauss3d:sigma", "sigma must be scalar or 3-vector");
    }

    // --- Parse kernel_size (optional) ---
    int ksize[3];
    if (nrhs >= 3 && !mxIsLogicalScalar(prhs[2])) {
        if (mxIsScalar(prhs[2])) {
            int k = (int)mxGetScalar(prhs[2]);
            ksize[0] = ksize[1] = ksize[2] = k;
        } else if (mxGetNumberOfElements(prhs[2]) == 3) {
            double* ks = mxGetPr(prhs[2]);
            for(int i=0; i<3; ++i) ksize[i] = (int)ks[i];
        } else {
            mexErrMsgIdAndTxt("gauss3d:kernel", "kernel_size must be scalar or 3-vector");
        }
    } else {
        for(int i=0; i<3; ++i)
            ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
    }

    // --- Check warning preference (4th argument, optional) ---
    int do_warn = 1;
    if (nrhs >= 4 && mxIsLogicalScalar(prhs[3])) {
        do_warn = !mxIsLogicalScalarTrue(prhs[3]);
    } else {
        // Check environment variable (for backward compatibility)
        const char* warn_env = std::getenv("GAUSS3D_WARN_KSIZE");
        if (warn_env && warn_env[0] == '0') do_warn = 0;
    }
    warn_kernel_size(sigma_double, ksize, do_warn);

    // --- Convert sigma to float or double as appropriate ---
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
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double");
    }

    // Return the modified input as output
    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
