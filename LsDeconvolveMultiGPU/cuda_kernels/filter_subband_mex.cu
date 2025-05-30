// filter_subband_mex.cu
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <vector>
#include <stdexcept>

#define THREADS_PER_BLOCK 256

__device__ float gaussian_notch(int i, float sigma, int n) {
    float x = float(i);
    return 1.0f - expf(-(x * x) / (2.0f * sigma * sigma));
}

// Apply notch filter in-place along rows or cols
__global__ void apply_filter_rows(float2* data, int rows, int cols, float sigma) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    for (int col = 0; col < cols; ++col) {
        int idx = row * cols + col;
        float g = gaussian_notch(col, sigma, cols);
        data[idx].x *= g;
        data[idx].y *= g;
    }
}

__global__ void apply_filter_cols(float2* data, int rows, int cols, float sigma) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    for (int row = 0; row < rows; ++row) {
        int idx = row * cols + col;
        float g = gaussian_notch(row, sigma, rows);
        data[idx].x *= g;
        data[idx].y *= g;
    }
}

__global__ void normalize_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] /= N;
    }
}


void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        mexErrMsgIdAndTxt("filter_subband_mex:CUDA", msg);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mxInitGPU();

    if (nrhs < 5) {
        mexErrMsgTxt("Usage: out = filter_subband_mex(img, sigma, levels, wavelet_code, axes)");
    }

    const mxGPUArray *imgGPU = mxGPUCreateFromMxArray(prhs[0]);

    // Get device ID from gpuArray pointer
    const void* voidPtr = mxGPUGetDataReadOnly(imgGPU);
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, voidPtr);
    if (err != cudaSuccess) {
        mexErrMsgIdAndTxt("filter_subband_mex:PointerAttr", cudaGetErrorString(err));
    }

    mexPrintf("CUDA device ID from input gpuArray: %d\n", attr.device);

    err = cudaSetDevice(attr.device);
    if (err != cudaSuccess) {
        mexErrMsgIdAndTxt("filter_subband_mex:SetDevice", cudaGetErrorString(err));
    }

    // Ensure CUDA context is initialized
    cudaFree(0);

    float sigma = mxGetScalar(prhs[1]);
    int levels = int(mxGetScalar(prhs[2]));
    int wavelet_code = int(mxGetScalar(prhs[3]));
    float *axes = (float*)mxGetData(prhs[4]);

    const mwSize *dims = mxGPUGetDimensions(imgGPU);
    int rows = dims[0];
    int cols = dims[1];

    // Create output
    mxGPUArray *outGPU = mxGPUCreateGPUArray(2, dims, mxSINGLE_CLASS,
                                             mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float *outData = static_cast<float*>(mxGPUGetData(outGPU));

    // Copy input to output buffer (in-place processing)
    cudaMemcpy(outData, mxGPUGetDataReadOnly(imgGPU), sizeof(float) * rows * cols, cudaMemcpyDeviceToDevice);

    // Allocate complex buffer for FFT
    cufftComplex *fft_data;
    cudaMalloc(&fft_data, sizeof(cufftComplex) * rows * cols);
    cudaMemcpy(fft_data, outData, sizeof(float) * rows * cols, cudaMemcpyDeviceToDevice);

    cufftHandle plan_fwd, plan_inv;
    cufftPlan2d(&plan_fwd, rows, cols, CUFFT_R2C);
    cufftPlan2d(&plan_inv, rows, cols, CUFFT_C2R);

    // Apply FFT
    cufftExecR2C(plan_fwd, outData, fft_data);

    // Apply filters
    if ((int)axes[0] == 1) {
        int blocks = (cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        apply_filter_cols<<<blocks, THREADS_PER_BLOCK>>>(fft_data, rows, cols, sigma);
    }
    if ((int)axes[0] == 2) {
        int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        apply_filter_rows<<<blocks, THREADS_PER_BLOCK>>>(fft_data, rows, cols, sigma);
    }

    // Inverse FFT
    cufftExecC2R(plan_inv, fft_data, outData);

    // Normalize result (cufft output needs to be divided by N)
    int N = rows * cols;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(outData, N);
    cudaDeviceSynchronize();

    // Output
    plhs[0] = mxGPUCreateMxArrayOnGPU(outGPU);

    // Cleanup
    cudaFree(fft_data);
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    mxGPUDestroyGPUArray(imgGPU);
    mxGPUDestroyGPUArray(outGPU);
}
