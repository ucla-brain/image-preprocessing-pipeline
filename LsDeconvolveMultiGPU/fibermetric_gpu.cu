#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring> // mxGetString
#include <stdexcept>

// ────────────────────── CONSTANTS AND MACROS ──────────────────────────────
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

constexpr float GAUSSIAN_KERNEL_HALFWIDTH_MULT = 4.0f; // Truncate Gaussian at ±4σ
constexpr double FINITE_DIFF_5PT_DIVISOR = 12.0;       // Δx = 1, used in 5-pt Laplacian
constexpr int THREADS_PER_BLOCK = 256;

// CUDA error checking macro
#define cudaCheck(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
            mexErrMsgIdAndTxt("fibermetric_gpu:cuda", \
                "CUDA error %s (%d) @ %s:%d", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
    } while (0)

// ────────────────────────── 3D INDEXING HELPERS ───────────────────────────
__device__ __host__ __forceinline__
int linearIndex3D(int row, int col, int slice, int numRows, int numCols) noexcept {
    return row + col * numRows + slice * numRows * numCols;
}

// ────────────── 1-D GAUSSIAN & FIRST DERIVATIVE KERNEL MAKER ──────────────
static inline void buildGaussianAndFirstDerivativeKernels(
    std::vector<double>& gaussianKernel,
    std::vector<double>& firstDerivativeKernel,
    double sigma)
{
    // Truncate at ±HALFWIDTH_MULT * σ
    const int halfWidth = static_cast<int>(std::ceil(GAUSSIAN_KERNEL_HALFWIDTH_MULT * sigma));
    const int kernelLen = 2 * halfWidth + 1;
    const double sigmaSq = sigma * sigma;
    const double norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);

    gaussianKernel.resize(kernelLen);
    firstDerivativeKernel.resize(kernelLen);

    double sumG = 0.0;
    for (int i = 0; i < kernelLen; ++i) {
        const int x = i - halfWidth;
        const double g = std::exp(-0.5 * x * x / sigmaSq) * norm;
        gaussianKernel[i] = g;
        firstDerivativeKernel[i] = (-x * g) / sigmaSq;
        sumG += g;
    }
    for (double& v : gaussianKernel) v /= sumG; // normalize so sum = 1
}

// ──────────────── SEPARABLE 1D CONVOLUTION ALONG AN AXIS ──────────────────
template<int AXIS>
__global__ void separableConvolution1DKernel(
    const float*  __restrict__ input,
    float*        __restrict__ output,
    const double* __restrict__ kernel,
    int kernelLen, int nRows, int nCols, int nSlices)
{
    const int halfWidth = kernelLen >> 1;
    int row   = (AXIS == 0) ? blockIdx.x * blockDim.x + threadIdx.x : blockIdx.x;
    int col   = (AXIS == 1) ? blockIdx.y * blockDim.y + threadIdx.y : blockIdx.y;
    int slice = (AXIS == 2) ? blockIdx.z * blockDim.z + threadIdx.z : blockIdx.z;
    if (row >= nRows || col >= nCols || slice >= nSlices) return;

    double acc = 0.0;
    for (int k = -halfWidth; k <= halfWidth; ++k) {
        int rr = row, cc = col, ss = slice;
        if (AXIS == 0) rr += k;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;
        // Reflective BCs
        rr = (rr < 0) ? -rr : (rr >= nRows   ? 2 * nRows   - rr - 2 : rr);
        cc = (cc < 0) ? -cc : (cc >= nCols   ? 2 * nCols   - cc - 2 : cc);
        ss = (ss < 0) ? -ss : (ss >= nSlices ? 2 * nSlices - ss - 2 : ss);

        acc += static_cast<double>(
            input[linearIndex3D(rr, cc, ss, nRows, nCols)]
        ) * kernel[k + halfWidth];
    }
    output[linearIndex3D(row, col, slice, nRows, nCols)] = static_cast<float>(acc);
}

static inline void launchSeparableConvolution(
    int axis,
    const float* srcDev, float* dstDev,
    const std::vector<double>& kernelHost, int kernelLen,
    int nRows, int nCols, int nSlices)
{
    // Copy kernel to device (always < 1kB)
    double* kernelDev = nullptr;
    cudaCheck(cudaMalloc(&kernelDev, kernelLen * sizeof(double)));
    cudaCheck(cudaMemcpy(kernelDev, kernelHost.data(), kernelLen * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blockDim(1,1,1), gridDim(nRows, nCols, nSlices);
    if (axis == 0) { blockDim.x = 32; gridDim.x = (nRows + 31) / 32; }
    if (axis == 1) { blockDim.y = 32; gridDim.y = (nCols + 31) / 32; }
    if (axis == 2) { blockDim.z = 32; gridDim.z = (nSlices + 31) / 32; }

    if (axis == 0)
        separableConvolution1DKernel<0><<<gridDim, blockDim>>>(srcDev, dstDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else if (axis == 1)
        separableConvolution1DKernel<1><<<gridDim, blockDim>>>(srcDev, dstDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else
        separableConvolution1DKernel<2><<<gridDim, blockDim>>>(srcDev, dstDev, kernelDev, kernelLen, nRows, nCols, nSlices);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaFree(kernelDev));
}

// ──────────── 5-POINT SECOND DERIVATIVE (FINITE DIFF) KERNEL ──────────────
template<int AXIS>
__global__ void secondDerivative5ptKernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int nRows, int nCols, int nSlices)
{
    int row   = (AXIS == 0) ? blockIdx.x * blockDim.x + threadIdx.x : blockIdx.x;
    int col   = (AXIS == 1) ? blockIdx.y * blockDim.y + threadIdx.y : blockIdx.y;
    int slice = (AXIS == 2) ? blockIdx.z * blockDim.z + threadIdx.z : blockIdx.z;
    if (row >= nRows || col >= nCols || slice >= nSlices) return;

    int rr[5] = {row,row,row,row,row};
    int cc[5] = {col,col,col,col,col};
    int ss[5] = {slice,slice,slice,slice,slice};
    for (int k = 0; k < 5; ++k) {
        int delta = k - 2;
        if (AXIS == 0) rr[k] += delta;
        if (AXIS == 1) cc[k] += delta;
        if (AXIS == 2) ss[k] += delta;
        rr[k] = (rr[k] < 0) ? -rr[k] : (rr[k] >= nRows   ? 2 * nRows   - rr[k] - 2 : rr[k]);
        cc[k] = (cc[k] < 0) ? -cc[k] : (cc[k] >= nCols   ? 2 * nCols   - cc[k] - 2 : cc[k]);
        ss[k] = (ss[k] < 0) ? -ss[k] : (ss[k] >= nSlices ? 2 * nSlices - ss[k] - 2 : ss[k]);
    }
    float lap =
        -input[linearIndex3D(rr[0], cc[0], ss[0], nRows, nCols)]
        + 16.0f * input[linearIndex3D(rr[1], cc[1], ss[1], nRows, nCols)]
        - 30.0f * input[linearIndex3D(rr[2], cc[2], ss[2], nRows, nCols)]
        + 16.0f * input[linearIndex3D(rr[3], cc[3], ss[3], nRows, nCols)]
        - input[linearIndex3D(rr[4], cc[4], ss[4], nRows, nCols)];
    output[linearIndex3D(row, col, slice, nRows, nCols)] = lap / static_cast<float>(FINITE_DIFF_5PT_DIVISOR);
}

// ────────────── SCALE ARRAY IN-PLACE (SCALE-SPACE NORMALIZATION) ──────────
__global__ void scaleArrayInPlaceKernel(float* arrDev, size_t n, float scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arrDev[idx] *= scale;
}

// ────────────────── 3x3 SYMMETRIC EIGENVALUE SOLVER (CUDA) ────────────────
__device__ __host__ __forceinline__
void computeSymmetricEigenvalues3x3(
    float A11, float A22, float A33, float A12, float A13, float A23,
    double& l1, double& l2, double& l3) noexcept
{
    const double q = (A11 + A22 + A33) / 3.0;
    const double B11 = A11 - q, B22 = A22 - q, B33 = A33 - q;
    const double p2 = (B11*B11 + B22*B22 + B33*B33 + 2.0*(A12*A12 + A13*A13 + A23*A23)) / 6.0;
    const double p = sqrt(p2);

    if (p < 1e-15) { l1 = l2 = l3 = q; return; }

    const double C11 = B11 / p, C22 = B22 / p, C33 = B33 / p;
    const double C12 = A12 / p, C13 = A13 / p, C23 = A23 / p;

    const double detC = C11*(C22*C33 - C23*C23) - C12*(C12*C33 - C23*C13) + C13*(C12*C23 - C22*C13);
    const double r = fmax(fmin(detC * 0.5, 1.0), -1.0);
    const double phi = acos(r) / 3.0;

    const double x1 = q + 2.0 * p * cos(phi);
    const double x3 = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
    const double x2 = 3.0 * q - x1 - x3;

    // Return eigenvalues ordered by |λ| ascending.
    double absVal[3] = {fabs(x1), fabs(x2), fabs(x3)};
    int order[3] = {0,1,2};
    for (int i = 0; i < 2; ++i)
        for (int j = i+1; j < 3; ++j)
            if (absVal[i] > absVal[j]) { std::swap(absVal[i], absVal[j]); std::swap(order[i], order[j]); }
    const double vals[3] = {x1, x2, x3};
    l1 = vals[order[0]]; l2 = vals[order[1]]; l3 = vals[order[2]];
}

// ──────────────────── FRANGI VESSELNESS EVALUATION KERNEL ──────────────────
__global__ void vesselnessFrangiKernel(
    const float* __restrict__ Dxx, const float* __restrict__ Dyy, const float* __restrict__ Dzz,
    const float* __restrict__ Dxy, const float* __restrict__ Dxz, const float* __restrict__ Dyz,
    float* __restrict__ vesselness, size_t n,
    float alpha, float beta, float gamma, bool bright)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double l1, l2, l3;
    computeSymmetricEigenvalues3x3(
        Dxx[idx], Dyy[idx], Dzz[idx],
        Dxy[idx], Dxz[idx], Dyz[idx], l1, l2, l3);

    // Frangi polarity condition: for "bright" structures, the smallest 2 eigenvalues must be negative
    bool ok = bright ? (l2 < 0.0 && l3 < 0.0) : (l2 > 0.0 && l3 > 0.0);

    double v = 0.0;
    if (ok) {
        const double absL1 = fabs(l1), absL2 = fabs(l2), absL3 = fabs(l3);
        const double Ra = absL2 / absL3;
        const double Rb = absL1 / sqrt(absL2 * absL3);
        const double S2 = l1*l1 + l2*l2 + l3*l3;
        v = (1.0 - exp(-Ra*Ra / (2.0*alpha*alpha)))
          * exp(-Rb*Rb / (2.0*beta*beta))
          * (1.0 - exp(-S2 / (2.0*gamma*gamma)));
    }
    vesselness[idx] = static_cast<float>(v);
}

// ────────────── ELEMENTWISE MAX PROJECTION FOR MULTISCALE ────────────────
__global__ void elementwiseMaxKernel(const float* src, float* dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaxf(dst[idx], src[idx]);
}

// ────────────────────────────── MEX ENTRY ────────────────────────────────
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 8)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Usage: V = fibermetric_gpu(gpuArraySingle3D, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, 'bright'|'dark', structureSensitivity)");

    mxInitGPU();

    // --- Parse input ---
    const mxGPUArray* inputGPU = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(inputGPU) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(inputGPU) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type", "Input must be 3D single gpuArray.");

    const mwSize* dims = mxGPUGetDimensions(inputGPU);
    int nRows = static_cast<int>(dims[0]), nCols = static_cast<int>(dims[1]), nSlices = static_cast<int>(dims[2]);
    size_t nElements = size_t(nRows) * nCols * nSlices;
    const float* inputDev = static_cast<const float*>(mxGPUGetDataReadOnly(inputGPU));

    const double sigmaFrom = mxGetScalar(prhs[1]);
    const double sigmaTo   = mxGetScalar(prhs[2]);
    const double sigmaStep = mxGetScalar(prhs[3]);
    const float alpha = static_cast<float>(mxGetScalar(prhs[4]));
    const float beta  = static_cast<float>(mxGetScalar(prhs[5]));
    char polarityBuff[16]; mxGetString(prhs[6], polarityBuff, sizeof(polarityBuff));
    bool brightPolarity = (std::strcmp(polarityBuff, "bright") == 0);
    const double structureSensitivity = mxGetScalar(prhs[7]);

    // --- Output allocation ---
    mxGPUArray* outputGPU = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(outputGPU));
    cudaCheck(cudaMemset(outputDev, 0, nElements * sizeof(float)));

    // --- Allocate all temp buffers and reuse ---
    float *tmpDev1, *tmpDev2, *DxxDev, *DyyDev, *DzzDev, *DxyDev, *DxzDev, *DyzDev, *vesselTmpDev;
    size_t volBytes = nElements * sizeof(float);
    cudaCheck(cudaMalloc(&tmpDev1, volBytes));
    cudaCheck(cudaMalloc(&tmpDev2, volBytes));
    cudaCheck(cudaMalloc(&DxxDev, volBytes));
    cudaCheck(cudaMalloc(&DyyDev, volBytes));
    cudaCheck(cudaMalloc(&DzzDev, volBytes));
    cudaCheck(cudaMalloc(&DxyDev, volBytes));
    cudaCheck(cudaMalloc(&DxzDev, volBytes));
    cudaCheck(cudaMalloc(&DyzDev, volBytes));
    cudaCheck(cudaMalloc(&vesselTmpDev, volBytes));

    int blocks1D = int((nElements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // --- Build sigma list ---
    std::vector<float> sigmaList;
    for (float s = static_cast<float>(sigmaFrom); s <= static_cast<float>(sigmaTo) + 1e-6f; s += static_cast<float>(sigmaStep))
        sigmaList.push_back(s);

    // --- Main multi-scale loop ---
    for (const float sigma : sigmaList)
    {
        // -- Build Gaussian and derivative kernels in double --
        std::vector<double> gaussHost, derivHost;
        buildGaussianAndFirstDerivativeKernels(gaussHost, derivHost, static_cast<double>(sigma));
        int kernelLen = int(gaussHost.size());

        // -- Gaussian smoothing (separable, 3 passes) --
        launchSeparableConvolution(0, inputDev, tmpDev1, gaussHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(1, tmpDev1, tmpDev2, gaussHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(2, tmpDev2, tmpDev1, gaussHost, kernelLen, nRows, nCols, nSlices);

        // -- Dxx, Dyy, Dzz (5pt) --
        dim3 bx(32,1,1), gx((nRows+31)/32, nCols, nSlices);
        secondDerivative5ptKernel<0><<<gx, bx>>>(tmpDev1, DxxDev, nRows, nCols, nSlices);

        dim3 by(1,32,1), gy(nRows, (nCols+31)/32, nSlices);
        secondDerivative5ptKernel<1><<<gy, by>>>(tmpDev1, DyyDev, nRows, nCols, nSlices);

        dim3 bz(1,1,32), gz(nRows, nCols, (nSlices+31)/32);
        secondDerivative5ptKernel<2><<<gz, bz>>>(tmpDev1, DzzDev, nRows, nCols, nSlices);

        cudaCheck(cudaGetLastError());

        // -- Dxy: ∂²/∂x∂y = d/dx then d/dy (1st deriv kernel, then 1st deriv kernel), then smooth in z --
        launchSeparableConvolution(0, inputDev, tmpDev1, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(1, tmpDev1, tmpDev2, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(2, tmpDev2, DxyDev, gaussHost, kernelLen, nRows, nCols, nSlices);

        // -- Dxz: d/dx then d/dz, then smooth in y --
        launchSeparableConvolution(0, inputDev, tmpDev1, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(2, tmpDev1, tmpDev2, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(1, tmpDev2, DxzDev, gaussHost, kernelLen, nRows, nCols, nSlices);

        // -- Dyz: d/dy then d/dz, then smooth in x --
        launchSeparableConvolution(1, inputDev, tmpDev1, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(2, tmpDev1, tmpDev2, derivHost, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(0, tmpDev2, DyzDev, gaussHost, kernelLen, nRows, nCols, nSlices);

        // -- Scale-space normalization (∂²I σ²) everywhere --
        float sigmaSq = sigma * sigma;
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DxxDev, nElements, sigmaSq);
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DyyDev, nElements, sigmaSq);
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DzzDev, nElements, sigmaSq);
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DxyDev, nElements, sigmaSq);
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DxzDev, nElements, sigmaSq);
        scaleArrayInPlaceKernel<<<blocks1D, THREADS_PER_BLOCK>>>(DyzDev, nElements, sigmaSq);

        cudaCheck(cudaGetLastError());

        // -- Vesselness computation --
        float gamma = static_cast<float>(structureSensitivity * sigma);
        vesselnessFrangiKernel<<<blocks1D, THREADS_PER_BLOCK>>>(
            DxxDev, DyyDev, DzzDev, DxyDev, DxzDev, DyzDev,
            vesselTmpDev, nElements, alpha, beta, gamma, brightPolarity);
        cudaCheck(cudaGetLastError());

        // -- Max projection for multi-scale (if more than 1 scale) --
        if (sigmaList.size() > 1)
            elementwiseMaxKernel<<<blocks1D, THREADS_PER_BLOCK>>>(vesselTmpDev, outputDev, nElements);
        else
            cudaCheck(cudaMemcpy(outputDev, vesselTmpDev, volBytes, cudaMemcpyDeviceToDevice));

        cudaCheck(cudaGetLastError());
    }

    // --- Return and cleanup ---
    plhs[0] = mxGPUCreateMxArrayOnGPU(outputGPU);

    cudaFree(tmpDev1); cudaFree(tmpDev2); cudaFree(DxxDev); cudaFree(DyyDev); cudaFree(DzzDev);
    cudaFree(DxyDev); cudaFree(DxzDev); cudaFree(DyzDev); cudaFree(vesselTmpDev);
    mxGPUDestroyGPUArray(inputGPU);
    mxGPUDestroyGPUArray(outputGPU);
}
