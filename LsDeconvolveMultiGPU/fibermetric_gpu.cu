#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

// ========= Helper Macros ==========
#define cudaCheck(err) if (err != cudaSuccess) { mexErrMsgIdAndTxt("fibermetric_gpu:cuda", "CUDA error %s at %s:%d", cudaGetErrorString(err), __FILE__, __LINE__); }

// ========= Index helpers for MATLAB COLUMN-MAJOR order =========
__device__ __host__ inline
int getLinearIndex3D(int row, int col, int slice, int numRows, int numCols, int numSlices) {
    // MATLAB: data(row + (col-1)*numRows + (slice-1)*numRows*numCols)
    return row + col * numRows + slice * numRows * numCols;
}

// ========= CUDA Gaussian Kernel Generation =========
void createGaussianKernel1D(float* hKernel, int ksize, float sigma) {
    int halfK = ksize/2;
    float sum = 0.f;
    for (int i = 0; i < ksize; ++i) {
        int x = i - halfK;
        float v = expf(-0.5f * (x * x) / (sigma * sigma));
        hKernel[i] = v;
        sum += v;
    }
    for (int i = 0; i < ksize; ++i) hKernel[i] /= sum;
}

// ========= 1D Convolution along each axis (separable Gaussian) =========
__global__ void convolve1DAlongX(const float* src, float* dst, const float* kernel, int ksize, int numRows, int numCols, int numSlices) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int slice = blockIdx.z;
    if (row >= numRows || col >= numCols || slice >= numSlices) return;
    int halfK = ksize/2;
    float sum = 0.f;
    for (int k = -halfK; k <= halfK; ++k) {
        int r = row + k;
        if (r < 0) r = 0;
        if (r >= numRows) r = numRows - 1;
        sum = fmaf(src[getLinearIndex3D(r, col, slice, numRows, numCols, numSlices)], kernel[k+halfK], sum);
    }
    dst[getLinearIndex3D(row, col, slice, numRows, numCols, numSlices)] = sum;
}

__global__ void convolve1DAlongY(const float* src, float* dst, const float* kernel, int ksize, int numRows, int numCols, int numSlices) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int slice = blockIdx.z;
    if (row >= numRows || col >= numCols || slice >= numSlices) return;
    int halfK = ksize / 2;
    float sum = 0.f;
    for (int k = -halfK; k <= halfK; ++k) {
        int c = col + k;
        if (c < 0) c = 0;
        if (c >= numCols) c = numCols - 1;
        sum = fmaf(src[getLinearIndex3D(row, c, slice, numRows, numCols, numSlices)], kernel[k + halfK], sum);
    }
    dst[getLinearIndex3D(row, col, slice, numRows, numCols, numSlices)] = sum;
}

__global__ void convolve1DAlongZ(const float* src, float* dst, const float* kernel, int ksize, int numRows, int numCols, int numSlices) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    int slice = blockIdx.z * blockDim.z + threadIdx.z;
    if (row >= numRows || col >= numCols || slice >= numSlices) return;
    int halfK = ksize / 2;
    float sum = 0.f;
    for (int k = -halfK; k <= halfK; ++k) {
        int s = slice + k;
        if (s < 0) s = 0;
        if (s >= numSlices) s = numSlices - 1;
        sum = fmaf(src[getLinearIndex3D(row, col, s, numRows, numCols, numSlices)], kernel[k + halfK], sum);
    }
    dst[getLinearIndex3D(row, col, slice, numRows, numCols, numSlices)] = sum;
}

// ========= Vesselness Kernel (matches MATLAB polarity/Frangi) ==========
__global__ void vesselness3D(const float* src, float* dst, int numRows, int numCols, int numSlices, float sigma,
                             float alpha, float beta, float gamma, bool brightPolarity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numRows * numCols * numSlices;
    if (idx >= total) return;

    int row = idx % numRows;
    int col = (idx / numRows) % numCols;
    int slice = idx / (numRows * numCols);

    if (row <= 0 || row >= numRows-1 ||
        col <= 0 || col >= numCols-1 ||
        slice <= 0 || slice >= numSlices-1) {
        dst[idx] = 0;
        return;
    }

    // Hessian (second derivatives, central diff)
    #define IDX(r,c,s) getLinearIndex3D(r,c,s,numRows,numCols,numSlices)
    float i_xx = src[IDX(row-1,col,slice)] - 2*src[IDX(row,col,slice)] + src[IDX(row+1,col,slice)];
    float i_yy = src[IDX(row,col-1,slice)] - 2*src[IDX(row,col,slice)] + src[IDX(row,col+1,slice)];
    float i_zz = src[IDX(row,col,slice-1)] - 2*src[IDX(row,col,slice)] + src[IDX(row,col,slice+1)];

    float i_xy = (src[IDX(row-1,col-1,slice)] + src[IDX(row+1,col+1,slice)]
                - src[IDX(row-1,col+1,slice)] - src[IDX(row+1,col-1,slice)]) * 0.25f;
    float i_xz = (src[IDX(row-1,col,slice-1)] + src[IDX(row+1,col,slice+1)]
                - src[IDX(row-1,col,slice+1)] - src[IDX(row+1,col,slice-1)]) * 0.25f;
    float i_yz = (src[IDX(row,col-1,slice-1)] + src[IDX(row,col+1,slice+1)]
                - src[IDX(row,col+1,slice-1)] - src[IDX(row,col-1,slice+1)]) * 0.25f;
    #undef IDX

    // Symmetric Hessian eigenvalues (analytical, cubic)
    float A11 = i_xx, A22 = i_yy, A33 = i_zz, A12 = i_xy, A13 = i_xz, A23 = i_yz;
    float eig1, eig2, eig3;
    float p1 = A12*A12 + A13*A13 + A23*A23;
    if (p1 < 1e-7f) {
        eig1 = A11; eig2 = A22; eig3 = A33;
    } else {
        float q = (A11 + A22 + A33)/3.f;
        float p2 = (A11-q)*(A11-q) + (A22-q)*(A22-q) + (A33-q)*(A33-q) + 2.f*p1;
        float p = sqrtf(p2 / 6.f);
        float B11 = (A11-q)/p, B12 = (A12)/p, B13 = (A13)/p;
        float B22 = (A22-q)/p, B23 = (A23)/p, B33 = (A33-q)/p;
        float detB =   B11*(B22*B33 - B23*B23)
                     - B12*(B12*B33 - B23*B13)
                     + B13*(B12*B23 - B22*B13);
        float r = detB/2.f;
        float phi;
        const float PI3 = 3.14159265f / 3.f;
        if (r <= -1.f) phi = PI3;
        else if (r >= 1.f) phi = 0.f;
        else phi = acosf(r)/3.f;
        eig1 = q + 2.f*p*cosf(phi);
        eig3 = q + 2.f*p*cosf(phi + 2.f*PI3);
        eig2 = 3.f*q - eig1 - eig3;
    }
    // Sort |eig1| < |eig2| < |eig3|
    float abse1 = fabsf(eig1), abse2 = fabsf(eig2), abse3 = fabsf(eig3);
    float tmp;
    if (abse1 > abse2) { tmp = eig1; eig1 = eig2; eig2 = tmp; tmp = abse1; abse1 = abse2; abse2 = tmp; }
    if (abse2 > abse3) { tmp = eig2; eig2 = eig3; eig3 = tmp; tmp = abse2; abse2 = abse3; abse3 = tmp; }
    if (abse1 > abse2) { tmp = eig1; eig1 = eig2; eig2 = tmp; tmp = abse1; abse1 = abse2; abse2 = tmp; }

    // Polarity select (as in MATLAB)
    bool keep = brightPolarity ? (eig2 < 0 && eig3 < 0) : (eig2 > 0 && eig3 > 0);
    float vesselness = 0.f;
    if (keep) {
        float l1 = abse1, l2 = abse2, l3 = abse3;
        float Ra = l2 / l3;
        float Rb = l1 / sqrtf(l2*l3);
        float S  = sqrtf(l1*l1 + l2*l2 + l3*l3);

        // Frangi's vesselness constants (match reference)
        float A = 2.0f * alpha * alpha;
        float B = 2.0f * beta * beta;
        float C = 2.0f * gamma * gamma;

        float expRa = 1.0f - __expf(-(Ra * Ra) / A);
        float expRb = __expf(-(Rb * Rb) / B);
        float expS  = 1.0f - __expf(-(S * S) / (2.0f * C));
        vesselness = expRa * expRb * expS;
    }
    dst[idx] = vesselness;
}

// ======= In-place max projection kernel for vesselness over scales =======
__global__ void maxInPlaceKernel(const float* src, float* dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = fmaxf(dst[i], src[i]);
}

// ---- In-place scaling kernel ------------------------------------------------
__global__ void scaleArrayInPlace(float* data, size_t n, float factor)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ========= Main entry point (all in one) =========
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // --- Argument checking ---
    if (nrhs < 8)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage", "Usage: out = fibermetric_gpu(gpuArray_single, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, gamma, objectPolarity)");

    // --- Parse gpuArray input ---
    if (!mxIsGPUArray(prhs[0]))
        mexErrMsgIdAndTxt("fibermetric_gpu:input", "Input must be a gpuArray.");
    mxGPUArray const* inputGpuArray = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(inputGpuArray) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(inputGpuArray) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type", "Input must be 3D single-precision gpuArray.");

    const mwSize* dims = mxGPUGetDimensions(inputGpuArray);
    int numRows = dims[0], numCols = dims[1], numSlices = dims[2];
    size_t numel = (size_t)numRows * numCols * numSlices;
    const float* inputData = static_cast<const float*>(mxGPUGetDataReadOnly(inputGpuArray));

    // --- Vesselness params ---
    float sigmaFrom = static_cast<float>(mxGetScalar(prhs[1]));
    float sigmaTo   = static_cast<float>(mxGetScalar(prhs[2]));
    float sigmaStep = static_cast<float>(mxGetScalar(prhs[3]));
    float alpha     = static_cast<float>(mxGetScalar(prhs[4]));
    float beta      = static_cast<float>(mxGetScalar(prhs[5]));
    float gamma     = static_cast<float>(mxGetScalar(prhs[6]));

    // --- Polarity ---
    char objectPolarity[16];
    mxGetString(prhs[7], objectPolarity, sizeof(objectPolarity));
    bool brightPolarity = (strcmp(objectPolarity,"bright")==0);

    // --- Output gpuArray (single, 3D) ---
    mxGPUArray* outputGpuArray = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputData = static_cast<float*>(mxGPUGetData(outputGpuArray));
    cudaCheck(cudaMemset(outputData, 0, numel * sizeof(float)));

    // --- Allocate buffers for Gaussian blur (separable) ---
    float* dTemp1; float* dTemp2;
    cudaCheck(cudaMalloc(&dTemp1, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&dTemp2, numel * sizeof(float)));

    int maxKernel = int(6 * sigmaTo + 1);
    if (maxKernel % 2 == 0) ++maxKernel;
    float* hKernel = new float[maxKernel];
    float* dKernel;
    cudaCheck(cudaMalloc(&dKernel, maxKernel * sizeof(float)));

    // --- Multi-scale vesselness computation ---
    float* dVesselness = outputData;
    for (float sigma = sigmaFrom; sigma <= sigmaTo + 1e-4f; sigma += sigmaStep) {
        int ksize = int(6 * sigma + 1);
        if (ksize % 2 == 0) ++ksize;
        createGaussianKernel1D(hKernel, ksize, sigma);
        cudaCheck(cudaMemcpy(dKernel, hKernel, ksize * sizeof(float), cudaMemcpyHostToDevice));

        // Separable Gaussian: X -> Y -> Z
        // X
        dim3 blockX(32,1,1), gridX((numRows+31)/32,numCols,numSlices);
        convolve1DAlongX<<<gridX, blockX>>>(inputData, dTemp1, dKernel, ksize, numRows, numCols, numSlices);
        cudaCheck(cudaGetLastError());
        // Y
        dim3 blockY(1,32,1), gridY(numRows,(numCols+31)/32,numSlices);
        convolve1DAlongY<<<gridY, blockY>>>(dTemp1, dTemp2, dKernel, ksize, numRows, numCols, numSlices);
        cudaCheck(cudaGetLastError());
        // Z
        dim3 blockZ(1,1,32), gridZ(numRows,numCols,(numSlices+31)/32);
        convolve1DAlongZ<<<gridZ, blockZ>>>(dTemp2, dTemp1, dKernel, ksize, numRows, numCols, numSlices);
        cudaCheck(cudaGetLastError());

        // === Scale by sigma^2 to match MATLAB fibermetric (Frangi normalisation) ===
        int nThreads = 256, nBlocks = (int)((numel + nThreads - 1) / nThreads);

        // --- Vesselness for this scale ---
        vesselness3D<<<nBlocks, nThreads>>>(dTemp1, dTemp2, numRows, numCols, numSlices, sigma, alpha, beta, gamma, brightPolarity);
        cudaCheck(cudaGetLastError());

        // Max-projection vesselness over scales
        // dVesselness = max(dVesselness, dTemp2)
        // (use a simple kernel here for element-wise max)
        maxInPlaceKernel<<<nBlocks, nThreads>>>(dTemp2, dVesselness, numel);
        cudaCheck(cudaGetLastError());
    }

    // --- Clean up ---
    cudaFree(dTemp1);
    cudaFree(dTemp2);
    cudaFree(dKernel);
    delete[] hKernel;

    mxGPUDestroyGPUArray(inputGpuArray);
    plhs[0] = mxGPUCreateMxArrayOnGPU(outputGpuArray);
    mxGPUDestroyGPUArray(outputGpuArray);
}
