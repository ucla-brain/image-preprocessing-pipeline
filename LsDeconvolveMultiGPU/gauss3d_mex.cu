// gauss3d_mex.cu - Minimal vRAM, robust, 3D Gaussian (separable) for MATLAB GPU

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

// Minimal vRAM version: only two buffers for in-place swapping.
template<typename T>
void run_gauss3d(T* bufA, T* bufB, int nx, int ny, int nz, float sigma[3]) {
    float *h_kernel = new float[MAX_KERNEL_SIZE];
    float *d_kernel;
    int klen;

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    // Axis order: X (0), Y (1), Z (2)
    T *src = bufA;
    T *dst = bufB;
    for (int dim = 0; dim < 3; ++dim) {
        make_gaussian_kernel(sigma[dim], h_kernel, &klen);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(float), cudaMemcpyHostToDevice));

        gauss1d_kernel<T><<<grid, block>>>(src, dst, nx, ny, nz, d_kernel, klen, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));

        // Swap pointers for next pass
        T* tmp = src;
        src = dst;
        dst = tmp;
    }
    delete[] h_kernel;
    // After three swaps, if passes is odd, src == bufB (final), else src == bufA.
    // We always return 'src' after 3 passes.
}

// MEX entry
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Need input array and sigma");

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
    mxGPUArray *A_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *B_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Copy input to A_gpu
    size_t N = (size_t)nx*ny*nz;
    void* input_ptr = (void*)mxGPUGetDataReadOnly(img_gpu);
    void* A_ptr = (void*)mxGPUGetData(A_gpu);
    CUDA_CHECK(cudaMemcpy(A_ptr, input_ptr, N * mxGPUGetElementSize(img_gpu), cudaMemcpyDeviceToDevice));

    void* B_ptr = (void*)mxGPUGetData(B_gpu);

    // Run filter (output may be in A or B)
    if (cls == mxSINGLE_CLASS)
        run_gauss3d<float>((float*)A_ptr, (float*)B_ptr, nx, ny, nz, sigma);
    else if (cls == mxDOUBLE_CLASS)
        run_gauss3d<double>((double*)A_ptr, (double*)B_ptr, nx, ny, nz, sigma);
    else mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double");

    // After 3 passes (odd), the output is in B_gpu
    plhs[0] = mxGPUCreateMxArrayOnGPU(B_gpu);

    mxGPUDestroyGPUArray(img_gpu);
    mxGPUDestroyGPUArray(A_gpu);
    mxGPUDestroyGPUArray(B_gpu);
}
