// gauss3d_mex.cu - 3D single-precision Gaussian filtering for MATLAB gpuArray
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// ========================
// CUDA kernel and helpers
// ========================
#define MAX_KERNEL_SIZE 51
__constant__ float const_kernel_f[MAX_KERNEL_SIZE];

void make_gaussian_kernel(float sigma, int ksize, float* kernel) {
    int r = ksize / 2;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
        sum += kernel[i + r];
    }
    for (int i = 0; i < ksize; ++i)
        kernel[i] = static_cast<float>(kernel[i] / sum);
}

__global__ void gauss1d_kernel_const_float(
    const float* __restrict__ src, float* __restrict__ dst,
    size_t nx, size_t ny, size_t nz,
    int klen, int axis)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nline, linelen;
    if (axis == 0) { linelen = nx; nline = ny * nz; }
    else if (axis == 1) { linelen = ny; nline = nx * nz; }
    else { linelen = nz; nline = nx * ny; }
    if (tid >= nline * linelen) return;

    size_t line = tid / linelen;
    size_t pos = tid % linelen;
    size_t x, y, z;
    if (axis == 0) { y = line % ny; z = line / ny; x = pos; }
    else if (axis == 1) { x = line % nx; z = line / nx; y = pos; }
    else { x = line % nx; y = line / nx; z = pos; }

    size_t idx = x + y * nx + z * nx * ny;
    int r = klen / 2;
    float acc = 0.0f;
    for (int s = 0; s < klen; ++s) {
        int offset = s - r;
        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);
        int zi = static_cast<int>(z);
        if (axis == 0) xi = min(max(static_cast<int>(x) + offset, 0), static_cast<int>(nx) - 1);
        if (axis == 1) yi = min(max(static_cast<int>(y) + offset, 0), static_cast<int>(ny) - 1);
        if (axis == 2) zi = min(max(static_cast<int>(z) + offset, 0), static_cast<int>(nz) - 1);
        size_t src_idx = xi + yi * nx + zi * nx * ny;
        acc += src[src_idx] * const_kernel_f[s];
    }
    dst[idx] = acc;
}

void gauss3d_separable_float(
    float* input,
    float* buffer,
    size_t nx, size_t ny, size_t nz,
    const float sigma[3], const int ksize[3])
{
    int max_klen = std::max(std::max(ksize[0], ksize[1]), ksize[2]);
    if (max_klen > MAX_KERNEL_SIZE) {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size exceeds MAX_KERNEL_SIZE (%d)", MAX_KERNEL_SIZE);
    }
    float* h_kernel = new float[max_klen];
    float* src = input;
    float* dst = buffer;

    int block = 256;
    for (int axis = 0; axis < 3; ++axis) {
        make_gaussian_kernel(sigma[axis], ksize[axis], h_kernel);
        cudaMemcpyToSymbol(const_kernel_f, h_kernel, ksize[axis] * sizeof(float), 0, cudaMemcpyHostToDevice);

        size_t linelen = (axis == 0) ? nx : (axis == 1) ? ny : nz;
        size_t nline   = (axis == 0) ? ny * nz : (axis == 1) ? nx * nz : nx * ny;
        size_t total = linelen * nline;
        int grid = static_cast<int>((total + block - 1) / block);

        gauss1d_kernel_const_float<<<grid, block, 0>>>(
            src, dst, nx, ny, nz, ksize[axis], axis);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            delete[] h_kernel;
            mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA kernel launch error: %s", cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            delete[] h_kernel;
            mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA synchronize error: %s", cudaGetErrorString(err));
        }
        std::swap(src, dst);
    }
    // After three passes, src is result (odd number of swaps)
    if (src != input) {
        cudaMemcpy(input, src, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    delete[] h_kernel;
}

// ==========================
// MEX entry point
// ==========================
extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2)
        mexErrMsgIdAndTxt("gauss3d:nrhs", "Usage: input_gpu = gauss3d_mex(input_gpu, buffer_gpu, sigma [, ksize])");

    // --- Parse input_gpu ---
    mxGPUArray* input_mx = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetNumberOfDimensions(input_mx) != 3 || mxGPUGetClassID(input_mx) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("gauss3d:input", "input_gpu must be a 3D single-precision gpuArray.");
    float* input_ptr = (float*)mxGPUGetData(input_mx);

    // --- Parse buffer_gpu ---
    mxGPUArray* buffer_mx = (mxGPUArray*)mxGPUCreateFromMxArray(prhs[1]);
    if (mxGPUGetNumberOfDimensions(buffer_mx) != 3 || mxGPUGetClassID(buffer_mx) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("gauss3d:buffer", "buffer_gpu must be a 3D single-precision gpuArray.");
    float* buffer_ptr = (float*)mxGPUGetData(buffer_mx);

    // --- Check size match ---
    const mwSize* sz_in = mxGPUGetDimensions(input_mx);
    const mwSize* sz_buf = mxGPUGetDimensions(buffer_mx);
    if (sz_in[0]!=sz_buf[0] || sz_in[1]!=sz_buf[1] || sz_in[2]!=sz_buf[2])
        mexErrMsgIdAndTxt("gauss3d:size", "input_gpu and buffer_gpu must have the same size.");

    // --- Parse sigma and ksize ---
    double sigma_d[3];
    if (nrhs > 2 && mxGetNumberOfElements(prhs[2])==3)
        for (int i=0; i<3; ++i) sigma_d[i] = mxGetPr(prhs[2])[i];
    else if (nrhs > 2 && mxIsScalar(prhs[2]))
        sigma_d[0] = sigma_d[1] = sigma_d[2] = mxGetScalar(prhs[2]);
    else
        mexErrMsgIdAndTxt("gauss3d:sigma", "sigma must be scalar or 3-vector.");

    int ksize[3];
    if (nrhs > 3 && mxGetNumberOfElements(prhs[3])==3)
        for (int i=0; i<3; ++i) ksize[i] = (int)mxGetPr(prhs[3])[i];
    else if (nrhs > 3 && mxIsScalar(prhs[3]))
        ksize[0] = ksize[1] = ksize[2] = (int)mxGetScalar(prhs[3]);
    else
        for (int i=0; i<3; ++i)
            ksize[i] = 2*(int)ceil(3.0*sigma_d[i])+1; // default from sigma

    // --- Run kernel orchestration ---
    float sigma[3] = { (float)sigma_d[0], (float)sigma_d[1], (float)sigma_d[2] };
    gauss3d_separable_float(input_ptr, buffer_ptr, sz_in[0], sz_in[1], sz_in[2], sigma, ksize);

    // --- Overwrite input_gpu with result (for MATLAB semantics) ---
    cudaMemcpy(input_ptr, buffer_ptr, sz_in[0]*sz_in[1]*sz_in[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    // --- Return modified input_gpu ---
    plhs[0] = mxGPUCreateMxArrayOnGPU(input_mx);
}
