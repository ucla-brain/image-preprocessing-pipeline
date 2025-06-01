// gauss3d_mex.cu - Robust 3D Gaussian convolution for MATLAB GPU (X->Y->Z, replicate padding)

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#define MAX_KERNEL_SIZE 51

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error: %s", cudaGetErrorString(err)); \
} while(0)

// CUDA kernel for 1D convolution along specified dimension (with replicate padding)
template <typename T>
__global__ void gauss1d_kernel(
    const T* src, T* dst, int nx, int ny, int nz,
    const float* kernel, int klen, int dim
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
        // Replicate (clamp) padding
        ci = min(max(ci, 0), size[dim] - 1);
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

// Host-side Gaussian kernel builder
void make_gaussian_kernel(float sigma, float* kernel, int* klen) {
    int r = (int)ceilf(3.0f * sigma);
    *klen = 2*r + 1;
    if (*klen > MAX_KERNEL_SIZE)
        mexErrMsgIdAndTxt("gauss3d:kernel", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        kernel[i+r] = expf(-0.5f * (i*i) / (sigma*sigma));
        sum += kernel[i+r];
    }
    for (int i = 0; i < *klen; ++i)
        kernel[i] /= sum;
}

// Separable 3D convolution: always returns the result in d_tmp
template<typename T>
void run_gauss3d(T* d_img, T* d_tmp, T* d_out, int nx, int ny, int nz, float sigma[3]) {
    float *h_kernels[3];
    float *d_kernels[3];
    int klen[3];
    for (int d = 0; d < 3; ++d) {
        h_kernels[d] = new float[MAX_KERNEL_SIZE];
        make_gaussian_kernel(sigma[d], h_kernels[d], &klen[d]);
        CUDA_CHECK(cudaMalloc(&d_kernels[d], klen[d] * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernels[d], h_kernels[d], klen[d] * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    // Pass 1: X (src: d_img → dst: d_tmp)
    gauss1d_kernel<T><<<grid, block>>>(d_img, d_tmp, nx, ny, nz, d_kernels[0], klen[0], 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2: Y (src: d_tmp → dst: d_out)
    gauss1d_kernel<T><<<grid, block>>>(d_tmp, d_out, nx, ny, nz, d_kernels[1], klen[1], 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 3: Z (src: d_out → dst: d_tmp)
    gauss1d_kernel<T><<<grid, block>>>(d_out, d_tmp, nx, ny, nz, d_kernels[2], klen[2], 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int d = 0; d < 3; ++d) {
        CUDA_CHECK(cudaFree(d_kernels[d]));
        delete[] h_kernels[d];
    }
    // After Z pass, d_tmp contains the final result!
}

// Main entry point for MATLAB MEX
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Need input array and sigma");

    // Accept both cpu and gpuArray input
    const mxGPUArray *img_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3) mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D");

    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];

    float sigma[3];
    if (mxIsScalar(prhs[1])) sigma[0]=sigma[1]=sigma[2]=(float)mxGetScalar(prhs[1]);
    else if (mxGetNumberOfElements(prhs[1])==3) {
        double* ps = mxGetPr(prhs[1]);
        for(int i=0;i<3;i++) sigma[i]=(float)ps[i];
    } else mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma must be scalar or length-3 vector");

    mxClassID cls = mxGPUGetClassID(img_gpu);
    mxGPUArray *out_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    void *img_ptr = (void*)mxGPUGetDataReadOnly(img_gpu);
    void *out_ptr = (void*)mxGPUGetData(out_gpu);

    mxGPUArray *tmp_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    void *tmp_ptr = (void*)mxGPUGetData(tmp_gpu);

    // Run: X (img->tmp), Y (tmp->out), Z (out->tmp). Final result in tmp.
    if (cls == mxSINGLE_CLASS)
        run_gauss3d<float>((float*)img_ptr, (float*)tmp_ptr, (float*)out_ptr, nx, ny, nz, sigma);
    else if (cls == mxDOUBLE_CLASS)
        run_gauss3d<double>((double*)img_ptr, (double*)tmp_ptr, (double*)out_ptr, nx, ny, nz, sigma);
    else mexErrMsgIdAndTxt("gauss3d:class", "Input
