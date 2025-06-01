// gauss3d_mex.cu
#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

#define MAX_KERNEL_SIZE 51  // reasonable max

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
        // Clamp to edge
        if (ci < 0) ci = 0;
        if (ci >= size[dim]) ci = size[dim] - 1;
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
    // support at least ±3*sigma, always odd kernel
    int r = (int)ceilf(3.0f * sigma);
    *klen = 2*r + 1;
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        kernel[i+r] = expf(-0.5f * (i*i) / (sigma*sigma));
        sum += kernel[i+r];
    }
    for (int i = 0; i < *klen; ++i)
        kernel[i] /= sum;
}

template<typename T>
void run_gauss3d(const T* d_img, T* d_tmp, T* d_out, int nx, int ny, int nz, float sigma[3]) {
    float *h_kernels[3];
    float *d_kernels[3];
    int klen[3];
    for (int d = 0; d < 3; ++d) {
        h_kernels[d] = new float[MAX_KERNEL_SIZE];
        make_gaussian_kernel(sigma[d], h_kernels[d], &klen[d]);
        cudaMalloc(&d_kernels[d], klen[d] * sizeof(float));
        cudaMemcpy(d_kernels[d], h_kernels[d], klen[d] * sizeof(float), cudaMemcpyHostToDevice);
    }

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    // x axis
    gauss1d_kernel<T><<<grid, block>>>(d_img, d_tmp, nx, ny, nz, d_kernels[0], klen[0], 0);
    cudaDeviceSynchronize();

    // y axis
    gauss1d_kernel<T><<<grid, block>>>(d_tmp, d_out, nx, ny, nz, d_kernels[1], klen[1], 1);
    cudaDeviceSynchronize();

    // z axis
    gauss1d_kernel<T><<<grid, block>>>(d_out, d_tmp, nx, ny, nz, d_kernels[2], klen[2], 2);
    cudaDeviceSynchronize();

    // copy result back to d_out
    cudaMemcpy(d_out, d_tmp, nx*ny*nz*sizeof(T), cudaMemcpyDeviceToDevice);

    for (int d = 0; d < 3; ++d) {
        cudaFree(d_kernels[d]);
        delete[] h_kernels[d];
    }
}

// MEX entry
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2) mexErrMsgIdAndTxt("gauss3d:nrhs", "Need input array and sigma");
    if (mxGetNumberOfDimensions(prhs[0]) != 3)
        mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D");

    const mwSize* sz = mxGetDimensions(prhs[0]);
    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];

    float sigma[3];
    if (mxIsScalar(prhs[1])) sigma[0]=sigma[1]=sigma[2]=(float)mxGetScalar(prhs[1]);
    else if (mxGetNumberOfElements(prhs[1])==3) {
        double* ps = mxGetPr(prhs[1]);
        for(int i=0;i<3;i++) sigma[i]=(float)ps[i];
    } else mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma must be scalar or length-3 vector");

    mxClassID cls = mxGetClassID(prhs[0]);
    mxArray* out = mxCreateNumericArray(3, sz, cls, mxREAL);
    plhs[0] = out;

    size_t N = nx*ny*nz;
    if (cls == mxSINGLE_CLASS) {
        const float* img = (float*)mxGetData(prhs[0]);
        float* d_img; float* d_tmp; float* d_out;
        cudaMalloc(&d_img, N*sizeof(float));
        cudaMalloc(&d_tmp, N*sizeof(float));
        cudaMalloc(&d_out, N*sizeof(float));
        cudaMemcpy(d_img, img, N*sizeof(float), cudaMemcpyHostToDevice);

        run_gauss3d<float>(d_img, d_tmp, d_out, nx, ny, nz, sigma);

        cudaMemcpy(mxGetData(out), d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_img); cudaFree(d_tmp); cudaFree(d_out);
    }
    else if (cls == mxDOUBLE_CLASS) {
        const double* img = (double*)mxGetData(prhs[0]);
        double* d_img; double* d_tmp; double* d_out;
        cudaMalloc(&d_img, N*sizeof(double));
        cudaMalloc(&d_tmp, N*sizeof(double));
        cudaMalloc(&d_out, N*sizeof(double));
        cudaMemcpy(d_img, img, N*sizeof(double), cudaMemcpyHostToDevice);

        run_gauss3d<double>(d_img, d_tmp, d_out, nx, ny, nz, sigma);

        cudaMemcpy(mxGetData(out), d_out, N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_img); cudaFree(d_tmp); cudaFree(d_out);
    } else
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double");
}
