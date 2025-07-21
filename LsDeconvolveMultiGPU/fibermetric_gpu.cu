#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

#define M_PI 3.14159265358979323846
// ========= Helper Macros ==========
#define cudaCheck(err) if (err != cudaSuccess) { mexErrMsgIdAndTxt("fibermetric_gpu:cuda", "CUDA error %s at %s:%d", cudaGetErrorString(err), __FILE__, __LINE__); }

void makeGaussianKernels1D(
    std::vector<float>& G, std::vector<float>& Gx, std::vector<float>& Gxx, int ksize, float sigma)
{
    int halfK = ksize / 2;
    float sumG = 0.f;
    float sigma2 = sigma * sigma, sigma4 = sigma2 * sigma2;
    float denom = sqrtf(2.0f * M_PI) * sigma;

    G.resize(ksize);
    Gx.resize(ksize);
    Gxx.resize(ksize);

    for (int i = 0; i < ksize; ++i) {
        int x = i - halfK;
        float g = expf(-0.5f * (x * x) / sigma2) / denom;
        G[i]   = g;
        Gx[i]  = -x * g / sigma2;
        Gxx[i] = (x * x - sigma2) * g / sigma4;
        sumG  += G[i];
    }
    for (int i = 0; i < ksize; ++i) G[i] /= sumG; // Normalize Gaussian
}

__device__ __host__ inline
int getLinearIndex3D(int row, int col, int slice, int numRows, int numCols, int numSlices) {
    return row + col * numRows + slice * numRows * numCols;
}

__global__ void convolve1DAlongX(const float* src, float* dst, const float* kernel, int ksize,
                                 int numRows, int numCols, int numSlices) {
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

__global__ void convolve1DAlongY(const float* src, float* dst, const float* kernel, int ksize,
                                 int numRows, int numCols, int numSlices) {
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

__global__ void convolve1DAlongZ(const float* src, float* dst, const float* kernel, int ksize,
                                 int numRows, int numCols, int numSlices) {
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

void launchConvX(const float* src, float* dst, const float* kernel, int ksize,
                 int numRows, int numCols, int numSlices) {
    dim3 block(32,1,1), grid((numRows+31)/32,numCols,numSlices);
    convolve1DAlongX<<<grid, block>>>(src, dst, kernel, ksize, numRows, numCols, numSlices);
}

void launchConvY(const float* src, float* dst, const float* kernel, int ksize,
                 int numRows, int numCols, int numSlices) {
    dim3 block(1,32,1), grid(numRows,(numCols+31)/32,numSlices);
    convolve1DAlongY<<<grid, block>>>(src, dst, kernel, ksize, numRows, numCols, numSlices);
}

void launchConvZ(const float* src, float* dst, const float* kernel, int ksize,
                 int numRows, int numCols, int numSlices) {
    dim3 block(1,1,32), grid(numRows,numCols,(numSlices+31)/32);
    convolve1DAlongZ<<<grid, block>>>(src, dst, kernel, ksize, numRows, numCols, numSlices);
}


void hessian3d_gpu(
    const float* input_d, int numRows, int numCols, int numSlices, float sigma, int ksize,
    float* Dxx_d, float* Dyy_d, float* Dzz_d, float* Dxy_d, float* Dxz_d, float* Dyz_d,
    float* temp1_d, float* temp2_d)
{
    // (Use all the kernels and helpers from previous code...)

    // Generate kernels
    std::vector<float> hG, hGx, hGxx;
    makeGaussianKernels1D(hG, hGx, hGxx, ksize, sigma);
    float *dG, *dGx, *dGxx;
    cudaCheck(cudaMalloc(&dG,   ksize * sizeof(float)));
    cudaCheck(cudaMalloc(&dGx,  ksize * sizeof(float)));
    cudaCheck(cudaMalloc(&dGxx, ksize * sizeof(float)));
    cudaCheck(cudaMemcpy(dG,   hG.data(),   ksize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dGx,  hGx.data(),  ksize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dGxx, hGxx.data(), ksize * sizeof(float), cudaMemcpyHostToDevice));

    // Dxx: Gxx(x) * G(y) * G(z)
    launchConvX(input_d, temp1_d, dGxx, ksize, numRows, numCols, numSlices);
    launchConvY(temp1_d, temp2_d, dG, ksize, numRows, numCols, numSlices);
    launchConvZ(temp2_d, Dxx_d, dG, ksize, numRows, numCols, numSlices);

    // Dyy: G(x) * Gxx(y) * G(z)
    launchConvY(input_d, temp1_d, dGxx, ksize, numRows, numCols, numSlices);
    launchConvX(temp1_d, temp2_d, dG, ksize, numRows, numCols, numSlices);
    launchConvZ(temp2_d, Dyy_d, dG, ksize, numRows, numCols, numSlices);

    // Dzz: G(x) * G(y) * Gxx(z)
    launchConvZ(input_d, temp1_d, dGxx, ksize, numRows, numCols, numSlices);
    launchConvX(temp1_d, temp2_d, dG, ksize, numRows, numCols, numSlices);
    launchConvY(temp2_d, Dzz_d, dG, ksize, numRows, numCols, numSlices);

    // Dxy: Gx(x) * Gx(y) * G(z)
    launchConvX(input_d, temp1_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvY(temp1_d, temp2_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvZ(temp2_d, Dxy_d, dG, ksize, numRows, numCols, numSlices);

    // Dxz: Gx(x) * G(z) * Gx(z)
    launchConvX(input_d, temp1_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvZ(temp1_d, temp2_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvY(temp2_d, Dxz_d, dG, ksize, numRows, numCols, numSlices);

    // Dyz: G(x) * Gx(y) * Gx(z)
    launchConvY(input_d, temp1_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvZ(temp1_d, temp2_d, dGx, ksize, numRows, numCols, numSlices);
    launchConvX(temp2_d, Dyz_d, dG, ksize, numRows, numCols, numSlices);

    cudaFree(dG); cudaFree(dGx); cudaFree(dGxx);
}


// ========= Vesselness Kernel (matches MATLAB polarity/Frangi) ==========
__global__ void vesselness3D(
    const float* Dxx, const float* Dyy, const float* Dzz,
    const float* Dxy, const float* Dxz, const float* Dyz,
    float* vesselness, int numRows, int numCols, int numSlices,
    float alpha, float beta, float gamma, bool brightPolarity)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = (size_t)numRows * numCols * numSlices;
    if (idx >= numel)
        return;

    // --- Compose symmetric Hessian for this voxel ---
    float A11 = Dxx[idx];
    float A22 = Dyy[idx];
    float A33 = Dzz[idx];
    float A12 = Dxy[idx];
    float A13 = Dxz[idx];
    float A23 = Dyz[idx];

    // --- Compute eigenvalues analytically (Cardano's cubic formula, robust to symmetric matrices) ---
    // See also Frangi and Kroon's code
    float eig1, eig2, eig3;
    float p1 = fmaf(A12, A12, fmaf(A13, A13, A23 * A23));
    if (p1 < 1e-7f) {
        eig1 = A11; eig2 = A22; eig3 = A33;
    } else {
        float q = (A11 + A22 + A33) / 3.0f;
        float p2 = fmaf(2.0f, p1, fmaf(A11-q, A11-q, fmaf(A22-q, A22-q, (A33-q)*(A33-q))));
        float p = sqrtf(p2 / 6.0f);
        float B11 = (A11 - q) / p, B12 = A12 / p, B13 = A13 / p;
        float B22 = (A22 - q) / p, B23 = A23 / p, B33 = (A33 - q) / p;
        float detB = fmaf(B11, fmaf(B22, B33, -B23*B23),
                       fmaf(-B12, fmaf(B12, B33, -B23*B13),
                         B13 * fmaf(B12, B23, -B22*B13)));
        float r = detB / 2.0f;
        float phi;
        const float PI3 = M_PI / 3.0f;
        if (r <= -1.0f)      phi = PI3;
        else if (r >= 1.0f)  phi = 0.0f;
        else                 phi = acosf(r) / 3.0f;
        eig1 = fmaf(2.0f * p, cosf(phi), q);
        eig3 = fmaf(2.0f * p, cosf(phi + 2.0f * PI3), q);
        eig2 = fmaf(3.0f, q, -eig1 - eig3); // = 3*q - eig1 - eig3
    }

    // --- Sort: |eig1| <= |eig2| <= |eig3| ---
    float abse1 = fabsf(eig1), abse2 = fabsf(eig2), abse3 = fabsf(eig3);
    float tmp;
    if (abse1 > abse2) { tmp = eig1; eig1 = eig2; eig2 = tmp; tmp = abse1; abse1 = abse2; abse2 = tmp; }
    if (abse2 > abse3) { tmp = eig2; eig2 = eig3; eig3 = tmp; tmp = abse2; abse2 = abse3; abse3 = tmp; }
    if (abse1 > abse2) { tmp = eig1; eig1 = eig2; eig2 = tmp; tmp = abse1; abse1 = abse2; abse2 = tmp; }

    // --- Vesselness per Frangi ---
    bool keep = brightPolarity ? (eig2 < 0 && eig3 < 0) : (eig2 > 0 && eig3 > 0);
    float result = 0.f;
    if (keep) {
        float l1 = abse1, l2 = abse2, l3 = abse3;
        float Ra = l2 / l3;
        float Rb = l1 / sqrtf(l2 * l3);
        float S  = sqrtf(l1 * l1 + l2 * l2 + l3 * l3);

        float A = 2.0f * alpha * alpha;
        float B = 2.0f * beta  * beta;
        float C = 2.0f * gamma * gamma;

        float expRa = 1.0f - __expf(-(Ra * Ra) / A);
        float expRb =        __expf(-(Rb * Rb) / B);
        float expS  = 1.0f - __expf(-(S  * S ) / C);

        result = expRa * expRb * expS;
    }
    vesselness[idx] = result;
}

// ======= In-place max projection kernel for vesselness over scales =======
__global__ void maxInPlaceThresholdKernel(const float* src, float* dst, size_t n, float threshold)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = fmaxf(dst[i], src[i]);
    dst[i] = (v >= threshold) ? v : 0.0f;
}


// ========= Main entry point (all in one) =========
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 8)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage", "Usage: out = fibermetric_gpu(gpuArray_single, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, gamma, objectPolarity)");

    // --- Parse input ---
    if (!mxIsGPUArray(prhs[0]))
        mexErrMsgIdAndTxt("fibermetric_gpu:input", "Input must be a gpuArray.");
    mxGPUArray const* inputGpuArray = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(inputGpuArray) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(inputGpuArray) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type", "Input must be 3D single-precision gpuArray.");

    const mwSize* dims = mxGPUGetDimensions(inputGpuArray);
    int numRows = dims[0], numCols = dims[1], numSlices = dims[2];
    size_t numel = (size_t)numRows * numCols * numSlices;
    const float* inputData = static_cast<const float*>(mxGPUGetDataReadOnly(inputGpuArray));

    float sigmaFrom = static_cast<float>(mxGetScalar(prhs[1]));
    float sigmaTo   = static_cast<float>(mxGetScalar(prhs[2]));
    float sigmaStep = static_cast<float>(mxGetScalar(prhs[3]));
    float alpha     = static_cast<float>(mxGetScalar(prhs[4]));
    float beta      = static_cast<float>(mxGetScalar(prhs[5]));
    float gamma     = static_cast<float>(mxGetScalar(prhs[6]));

    char objectPolarity[16];
    mxGetString(prhs[7], objectPolarity, sizeof(objectPolarity));
    bool brightPolarity = (strcmp(objectPolarity,"bright")==0);

    // --- Output ---
    mxGPUArray* outputGpuArray = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputData = static_cast<float*>(mxGPUGetData(outputGpuArray));
    cudaCheck(cudaMemset(outputData, 0, numel * sizeof(float)));

    // --- Work buffers (for convolution, Hessian, vesselness) ---
    float *temp1_d, *temp2_d, *Dxx_d, *Dyy_d, *Dzz_d, *Dxy_d, *Dxz_d, *Dyz_d, *vesselness_d;
    cudaCheck(cudaMalloc(&temp1_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&temp2_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dxx_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dyy_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dzz_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dxy_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dxz_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&Dyz_d, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&vesselness_d, numel * sizeof(float)));

    int nThreads = 256, nBlocks = (int)((numel + nThreads - 1) / nThreads);

    // --- Multi-scale vesselness computation ---
    for (float sigma = sigmaFrom; sigma <= sigmaTo + 1e-4f; sigma += sigmaStep) {
        int ksize = int(ceil(6.0f * sigma));
        if (ksize % 2 == 0) ++ksize;

        // 1. Hessian of Gaussian at this scale
        hessian3d_gpu(
            inputData, numRows, numCols, numSlices, sigma, ksize,
            Dxx_d, Dyy_d, Dzz_d, Dxy_d, Dxz_d, Dyz_d,
            temp1_d, temp2_d
        );
        cudaCheck(cudaGetLastError());

        // 2. Vesselness (per Frangi) at this scale
        cudaCheck(cudaMemset(vesselness_d, 0, numel * sizeof(float)));
        vesselness3D<<<nBlocks, nThreads>>>(
            Dxx_d, Dyy_d, Dzz_d, Dxy_d, Dxz_d, Dyz_d,
            vesselness_d, numRows, numCols, numSlices,
            alpha, beta, gamma, brightPolarity
        );
        cudaCheck(cudaGetLastError());

        // 3. Max-projection across scales
        float structureSensitivity = 0.0f;
        if (nrhs >= 9) {
            structureSensitivity = static_cast<float>(mxGetScalar(prhs[8]));
        }
        maxInPlaceThresholdKernel<<<nBlocks, nThreads>>>(vesselness_d, outputData, numel, structureSensitivity);
        cudaCheck(cudaGetLastError());
    }

    // --- Clean up ---
    cudaFree(temp1_d); cudaFree(temp2_d);
    cudaFree(Dxx_d); cudaFree(Dyy_d); cudaFree(Dzz_d);
    cudaFree(Dxy_d); cudaFree(Dxz_d); cudaFree(Dyz_d);
    cudaFree(vesselness_d);

    mxGPUDestroyGPUArray(inputGpuArray);
    plhs[0] = mxGPUCreateMxArrayOnGPU(outputGpuArray);
    mxGPUDestroyGPUArray(outputGpuArray);
}