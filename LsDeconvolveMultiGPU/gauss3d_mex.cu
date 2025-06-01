#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cassert>

// Macro to check for CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

// Max kernel size (constant memory)
#define MAX_KERNEL_SIZE 51

__constant__ float  const_kernel_f[MAX_KERNEL_SIZE];
__constant__ double const_kernel_d[MAX_KERNEL_SIZE];

// Template to get the right const memory pointer at compile time
template <typename T>
struct KernelConstPtr;
template <>
struct KernelConstPtr<float> { static __device__ __forceinline__ const float* ptr() { return const_kernel_f; } };
template <>
struct KernelConstPtr<double> { static __device__ __forceinline__ const double* ptr() { return const_kernel_d; } };

// CUDA kernel for separable Gaussian filtering (templated for float/double)
template <typename T>
__global__ void gauss1d_kernel_const(
    const T* src, T* dst,
    int nx, int ny, int nz,
    int klen, int axis)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else               { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    int line = tid / linelen;
    int pos  = tid % linelen;

    int x, y, z;
    if (axis == 0) {
        y = line % ny; z = line / ny; x = pos;
    } else if (axis == 1) {
        x = line % nx; z = line / nx; y = pos;
    } else {
        x = line % nx; y = line / nx; z = pos;
    }

    int idx = x + y * nx + z * nx * ny;
    int r   = klen / 2;
    T acc   = (T)0;
    int N   = nx * ny * nz;

    if (idx < 0 || idx >= N) {
        printf("OOB-dst: idx=%d N=%d [axis=%d]\n", idx, N, axis);
        return;
    }

    const T* kernel = KernelConstPtr<T>::ptr();

    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = x, yi = y, zi = z;
        if (axis == 0) xi = min(max(x + offset, 0), nx - 1);
        if (axis == 1) yi = min(max(y + offset, 0), ny - 1);
        if (axis == 2) zi = min(max(z + offset, 0), nz - 1);
        int src_idx = xi + yi * nx + zi * nx * ny;
        if (src_idx < 0 || src_idx >= N) {
            printf("OOB-src: src_idx=%d N=%d [axis=%d s=%d]\n", src_idx, N, axis, s);
            return;
        }
        if (s < 0 || s >= klen) {
            printf("OOB-kernel: s=%d klen=%d [axis=%d]\n", s, klen, axis);
            return;
        }
        acc += src[src_idx] * kernel[s];
    }
    dst[idx] = acc;
}

// Gaussian kernel creation
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

// Host-side orchestration for single/double precision
template <typename T>
void gauss3d_separable(
    T* input,
    T* buffer,
    int nx, int ny, int nz,
    const T sigma[3], const int ksize[3])
{
    assert(nx > 0 && ny > 0 && nz > 0);
    int max_klen = std::max(std::max(ksize[0], ksize[1]), ksize[2]);
    if (max_klen > MAX_KERNEL_SIZE) {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    }
    T* h_kernel = new T[max_klen];

    T* src = input;
    T* dst = buffer;

    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);

        if (std::is_same<T, float>::value) {
            CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_f, h_kernel, ksize[axis] * sizeof(float), 0, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(const_kernel_d, h_kernel, ksize[axis] * sizeof(double), 0, cudaMemcpyHostToDevice));
        }

        int linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        int nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        int total   = linelen * nline;
        int block   = 256;
        int grid    = (total + block - 1) / block;

        gauss1d_kernel_const<T><<<grid, block, 0>>>(src, dst, nx, ny, nz, ksize[axis], axis);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(src, dst);
    }
    // After 3 axes, src points to the result (due to odd number of swaps)
    if (src != input) {
        CUDA_CHECK(cudaMemcpy(input, src, (size_t)nx * ny * nz * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    delete[] h_kernel;
}

// ================
// MEX entry point
// ================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2)
        mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: gauss3d_mex(x, sigma [, kernel_size])");

    const mxGPUArray* img_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3)
        mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D.");

    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];
    size_t N = (size_t)nx * ny * nz;
    mxClassID cls = mxGPUGetClassID(img_gpu);
    void* ptr = mxGPUGetData(const_cast<mxGPUArray*>(img_gpu));
    void* buffer = nullptr;

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
        if (mxIsEmpty(prhs[2])) {
            for (int i = 0; i < 3; ++i)
                ksize[i] = 2 * (int)ceil(3.0 * sigma_double[i]) + 1;
        } else if (mxIsScalar(prhs[2])) {
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

    if (cls == mxSINGLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(float)));
        float sigma[3] = { (float)sigma_double[0], (float)sigma_double[1], (float)sigma_double[2] };
        gauss3d_separable((float*)ptr, (float*)buffer, nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaFree(buffer));
    } else if (cls == mxDOUBLE_CLASS) {
        CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(double)));
        double sigma[3] = { sigma_double[0], sigma_double[1], sigma_double[2] };
        gauss3d_separable((double*)ptr, (double*)buffer, nx, ny, nz, sigma, ksize);
        CUDA_CHECK(cudaFree(buffer));
    } else {
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double gpuArray");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // Do not destroy img_gpu before return
}
