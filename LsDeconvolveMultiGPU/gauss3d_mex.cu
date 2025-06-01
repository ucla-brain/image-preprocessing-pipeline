// gauss3d_mex_inplace.cu: In-place 3D Gaussian filter for MATLAB gpuArray input (no extra VRAM)
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
    T* bufA, T* bufB, int nx, int ny, int nz,
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
        ci = min(max(ci, 0), size[dim] - 1);
        int cidx;
        if (dim == 0)
            cidx = iz * nx * ny + iy * nx + ci;
        else if (dim == 1)
            cidx = iz * nx * ny + ci * nx + ix;
        else
            cidx = ci * nx * ny + iy * nx + ix;
        val += bufA[cidx] * kernel[k];
    }
    bufB[idx] = val;
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

// Minimal VRAM: 1 buffer (input), 1 temp buffer.
template<typename T>
void run_gauss3d_inplace(T* bufA, int nx, int ny, int nz, float sigma[3]) {
    float *h_kernel = new float[MAX_KERNEL_SIZE];
    float *d_kernel;
    int klen;

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    // Allocate one temp buffer for swap
    T* bufB;
    CUDA_CHECK(cudaMalloc(&bufB, nx*ny*nz*sizeof(T)));

    T *src = bufA;
    T *dst = bufB;
    for (int dim = 0; dim < 3; ++dim) {
        make_gaussian_kernel(sigma[dim], h_kernel, &klen);
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(float), cudaMemcpyHostToDevice));

        gauss1d_kernel<T><<<grid, block>>>(src, dst, nx, ny, nz, d_kernel, klen, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));

        // Swap: after last pass, dst will be bufA if odd, bufB if even
        T* tmp = src; src = dst; dst = tmp;
    }

    // If output is not in bufA, copy back from bufB (odd number of passes swaps back)
    if (src != bufA) {
        CUDA_CHECK(cudaMemcpy(bufA, bufB, nx*ny*nz*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaFree(bufB));
    delete[] h_kernel;
}

// MEX entry (in-place modification of input gpuArray)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Need input array and sigma");

    // Use input gpuArray for in-place output
    mxGPUArray* img_gpu = mxGPUCreateFromMxArray(prhs[0]);
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
    void* ptr = mxGPUGetData(img_gpu);

    if (cls == mxSINGLE_CLASS)
        run_gauss3d_inplace<float>((float*)ptr, nx, ny, nz, sigma);
    else if (cls == mxDOUBLE_CLASS)
        run_gauss3d_inplace<double>((double*)ptr, nx, ny, nz, sigma);
    else mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double");

    // Return the (modified) input as output, **do not destroy img_gpu**
    plhs[0] = mxGPUCreateMxArrayOnGPU(img_gpu);
    // mxGPUDestroyGPUArray(img_gpu); // Do not destroy if returning as output
}
