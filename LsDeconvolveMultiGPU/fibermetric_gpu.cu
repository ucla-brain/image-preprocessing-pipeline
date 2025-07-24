#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

// ---------------------------- Constants & Macros ----------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr float GAUSS_HALFWIDTH_MULT = 8.0f;
constexpr double FINITE_DIFF_5PT_DIVISOR = 12.0;
constexpr int THREADS_PER_BLOCK = 256;

#define cudaCheck(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
            mexErrMsgIdAndTxt("fibermetric_gpu:cuda", \
                "CUDA error %s (%d) @ %s:%d", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
    } while (0)

// ---------------------------- Device Functions -----------------------------
__device__ __host__ __forceinline__
int linearIndex3D(int row, int col, int slice, int nRows, int nCols) noexcept {
    return row + col * nRows + slice * nRows * nCols;
}

template<int AXIS>
__global__ void separableConvolution1DKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const double* __restrict__ kernel,
    int kernelLen, int nRows, int nCols, int nSlices)
{
    int halfWidth = kernelLen >> 1;
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

        // Reflect at boundaries
        rr = (rr < 0) ? -rr : (rr >= nRows   ? 2 * nRows   - rr - 2 : rr);
        cc = (cc < 0) ? -cc : (cc >= nCols   ? 2 * nCols   - cc - 2 : cc);
        ss = (ss < 0) ? -ss : (ss >= nSlices ? 2 * nSlices - ss - 2 : ss);

        acc += static_cast<double>(input[linearIndex3D(rr, cc, ss, nRows, nCols)]) *
               kernel[k + halfWidth];
    }

    output[linearIndex3D(row, col, slice, nRows, nCols)] = static_cast<float>(acc);
}

void launchSeparableConvolution(
    int axis, const float* inputDev, float* outputDev,
    const std::vector<double>& kernelHost, int kernelLen,
    int nRows, int nCols, int nSlices)
{
    double* kernelDev = nullptr;
    cudaCheck(cudaMalloc(&kernelDev, kernelLen * sizeof(double)));
    cudaCheck(cudaMemcpy(kernelDev, kernelHost.data(), kernelLen * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blockDim(1, 1, 1), gridDim(nRows, nCols, nSlices);
    if (axis == 0) { blockDim.x = 32; gridDim.x = (nRows + 31) / 32; }
    if (axis == 1) { blockDim.y = 32; gridDim.y = (nCols + 31) / 32; }
    if (axis == 2) { blockDim.z = 32; gridDim.z = (nSlices + 31) / 32; }

    if (axis == 0)
        separableConvolution1DKernel<0><<<gridDim, blockDim>>>(inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else if (axis == 1)
        separableConvolution1DKernel<1><<<gridDim, blockDim>>>(inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else
        separableConvolution1DKernel<2><<<gridDim, blockDim>>>(inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaFree(kernelDev));
}

template<int AXIS>
__global__ void secondDerivative5ptKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int nRows, int nCols, int nSlices)
{
    int row   = (AXIS == 0) ? blockIdx.x * blockDim.x + threadIdx.x : blockIdx.x;
    int col   = (AXIS == 1) ? blockIdx.y * blockDim.y + threadIdx.y : blockIdx.y;
    int slice = (AXIS == 2) ? blockIdx.z * blockDim.z + threadIdx.z : blockIdx.z;

    if (row >= nRows || col >= nCols || slice >= nSlices) return;

    int rr[5] = {row, row, row, row, row};
    int cc[5] = {col, col, col, col, col};
    int ss[5] = {slice, slice, slice, slice, slice};

    for (int k = 0; k < 5; ++k) {
        int offset = k - 2;
        if (AXIS == 0) rr[k] += offset;
        if (AXIS == 1) cc[k] += offset;
        if (AXIS == 2) ss[k] += offset;

        rr[k] = (rr[k] < 0) ? -rr[k] : (rr[k] >= nRows   ? 2 * nRows   - rr[k] - 2 : rr[k]);
        cc[k] = (cc[k] < 0) ? -cc[k] : (cc[k] >= nCols   ? 2 * nCols   - cc[k] - 2 : cc[k]);
        ss[k] = (ss[k] < 0) ? -ss[k] : (ss[k] >= nSlices ? 2 * nSlices - ss[k] - 2 : ss[k]);
    }

    float value =
        -input[linearIndex3D(rr[0], cc[0], ss[0], nRows, nCols)]
        + 16.0f * input[linearIndex3D(rr[1], cc[1], ss[1], nRows, nCols)]
        - 30.0f * input[linearIndex3D(rr[2], cc[2], ss[2], nRows, nCols)]
        + 16.0f * input[linearIndex3D(rr[3], cc[3], ss[3], nRows, nCols)]
        - input[linearIndex3D(rr[4], cc[4], ss[4], nRows, nCols)];

    output[linearIndex3D(row, col, slice, nRows, nCols)] =
        value / static_cast<float>(FINITE_DIFF_5PT_DIVISOR);
}

void launchSecondDerivatives(
    const float* smoothedInput, float* Dxx, float* Dyy, float* Dzz,
    int nRows, int nCols, int nSlices)
{
    dim3 bx(32,1,1), gx((nRows+31)/32, nCols, nSlices);
    secondDerivative5ptKernel<0><<<gx, bx>>>(smoothedInput, Dxx, nRows, nCols, nSlices);

    dim3 by(1,32,1), gy(nRows, (nCols+31)/32, nSlices);
    secondDerivative5ptKernel<1><<<gy, by>>>(smoothedInput, Dyy, nRows, nCols, nSlices);

    dim3 bz(1,1,32), gz(nRows, nCols, (nSlices+31)/32);
    secondDerivative5ptKernel<2><<<gz, bz>>>(smoothedInput, Dzz, nRows, nCols, nSlices);

    cudaCheck(cudaGetLastError());
}

void launchCrossDerivatives(
    const float* inputDev,
    float* Dxy, float* Dxz, float* Dyz,
    const std::vector<double>& derivHost,
    const std::vector<double>& gaussHost,
    int kernelLen,
    int nRows, int nCols, int nSlices)
{
    float *tmp1, *tmp2;
    size_t bytes = size_t(nRows) * nCols * nSlices * sizeof(float);
    cudaCheck(cudaMalloc(&tmp1, bytes));
    cudaCheck(cudaMalloc(&tmp2, bytes));

    // Dxy
    launchSeparableConvolution(0, inputDev, tmp1, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(1, tmp1, tmp2, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(2, tmp2, Dxy, gaussHost, kernelLen, nRows, nCols, nSlices);

    // Dxz
    launchSeparableConvolution(0, inputDev, tmp1, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(2, tmp1, tmp2, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(1, tmp2, Dxz, gaussHost, kernelLen, nRows, nCols, nSlices);

    // Dyz
    launchSeparableConvolution(1, inputDev, tmp1, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(2, tmp1, tmp2, derivHost, kernelLen, nRows, nCols, nSlices);
    launchSeparableConvolution(0, tmp2, Dyz, gaussHost, kernelLen, nRows, nCols, nSlices);

    cudaFree(tmp1);
    cudaFree(tmp2);
}

template <typename T>
__device__ __host__ __forceinline__
void swapCUDA(T& a, T& b) noexcept
{
    T temp = a;
    a = b;
    b = temp;
}


// Robust 3x3 symmetric matrix eigenvalue solver
__device__ __host__ __forceinline__
void computeSymmetricEigenvalues3x3(
    float A11, float A22, float A33, float A12, float A13, float A23,
    double& l1, double& l2, double& l3) noexcept
{
    double q = (A11 + A22 + A33) / 3.0;
    double B11 = A11 - q, B22 = A22 - q, B33 = A33 - q;
    double p2 = (B11*B11 + B22*B22 + B33*B33 + 2.0*(A12*A12 + A13*A13 + A23*A23)) / 6.0;
    double p = sqrt(p2);
    if (p < 1e-15) { l1 = l2 = l3 = q; return; }

    double C11 = B11 / p, C22 = B22 / p, C33 = B33 / p;
    double C12 = A12 / p, C13 = A13 / p, C23 = A23 / p;

    double detC = C11*(C22*C33 - C23*C23) - C12*(C12*C33 - C23*C13) + C13*(C12*C23 - C22*C13);
    double r = fmax(fmin(detC * 0.5, 1.0), -1.0);
    double phi = acos(r) / 3.0;

    double x1 = q + 2.0 * p * cos(phi);
    double x3 = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
    double x2 = 3.0 * q - x1 - x3;

    double absVals[3] = {fabs(x1), fabs(x2), fabs(x3)};
    int order[3] = {0, 1, 2};
    if (absVals[0] > absVals[1]) { swapCUDA(order[0], order[1]); swapCUDA(absVals[0], absVals[1]); }
    if (absVals[1] > absVals[2]) { swapCUDA(order[1], order[2]); swapCUDA(absVals[1], absVals[2]); }
    if (absVals[0] > absVals[1]) { swapCUDA(order[0], order[1]); swapCUDA(absVals[0], absVals[1]); }

    const double vals[3] = {x1, x2, x3};
    l1 = vals[order[0]]; l2 = vals[order[1]]; l3 = vals[order[2]];
}

// Vesselness kernel
__global__ void vesselnessFrangiKernel(
    const float* __restrict__ Dxx, const float* __restrict__ Dyy, const float* __restrict__ Dzz,
    const float* __restrict__ Dxy, const float* __restrict__ Dxz, const float* __restrict__ Dyz,
    float* __restrict__ vesselness, size_t n,
    double alpha, double beta, double gamma, bool bright)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double l1, l2, l3;
    computeSymmetricEigenvalues3x3(
        Dxx[idx], Dyy[idx], Dzz[idx], Dxy[idx], Dxz[idx], Dyz[idx], l1, l2, l3);

    bool condition = bright ? (l2 < 0.0 && l3 < 0.0) : (l2 > 0.0 && l3 > 0.0);
    double response = 0.0;
    if (condition) {
        double absL1 = fabs(l1), absL2 = fabs(l2), absL3 = fabs(l3);
        double Ra = absL2 / absL3;
        double Rb = absL1 / sqrt(absL2 * absL3 + 1e-30);
        double S2 = l1*l1 + l2*l2 + l3*l3;
        response = (1.0 - exp(-Ra*Ra / (2.0*alpha*alpha)))
                 *        exp(-Rb*Rb / (2.0*beta*beta))
                 * (1.0 - exp(-S2    / (2.0*gamma*gamma)));
    }
    vesselness[idx] = static_cast<float>(response);
}

// Max-projection kernel across scales
__global__ void elementwiseMaxKernel(const float* src, float* dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaxf(dst[idx], src[idx]);
}

// In-place scale kernel
__global__ void scaleArrayInPlaceKernel(float* arr, size_t n, float factor)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] *= factor;
}

// --------------------------- Host Helper Functions -------------------------
static inline void buildGaussianAndFirstDerivativeKernels(
    std::vector<double>& gauss, std::vector<double>& deriv, double sigma)
{
    int half = int(std::ceil(GAUSS_HALFWIDTH_MULT * sigma));
    int len = 2 * half + 1;
    double sigma2 = sigma * sigma;
    double norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);

    gauss.resize(len);
    deriv.resize(len);
    double sum = 0.0;
    for (int i = 0; i < len; ++i) {
        int x = i - half;
        double g = exp(-0.5 * x * x / sigma2) * norm;
        gauss[i] = g;
        deriv[i] = (-x * g) / sigma2;
        sum += g;
    }
    for (double& v : gauss) v /= sum;
}

// --------------------------- Entry Point (MEX) -----------------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 8)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Usage: fibermetric_gpu(gpuArraySingle3D, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, 'bright'|'dark', structureSensitivity)");

    mxInitGPU();
    const mxGPUArray* input = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(input) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(input) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type", "Input must be 3D single gpuArray.");

    const mwSize* dims = mxGPUGetDimensions(input);
    int nRows = int(dims[0]), nCols = int(dims[1]), nSlices = int(dims[2]);
    size_t n = size_t(nRows) * nCols * nSlices;
    const float* inputDev = static_cast<const float*>(mxGPUGetDataReadOnly(input));

    double sigmaFrom = mxGetScalar(prhs[1]);
    double sigmaTo   = mxGetScalar(prhs[2]);
    double sigmaStep = mxGetScalar(prhs[3]);
    float alpha = float(mxGetScalar(prhs[4]));
    float beta  = float(mxGetScalar(prhs[5]));
    char polarityBuf[16]; mxGetString(prhs[6], polarityBuf, sizeof(polarityBuf));
    bool bright = std::strcmp(polarityBuf, "bright") == 0;
    double structureSensitivity = mxGetScalar(prhs[7]);

    mxGPUArray* output = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(output));
    cudaCheck(cudaMemset(outputDev, 0, n * sizeof(float)));

    float *tmp1, *tmp2, *Dxx, *Dyy, *Dzz, *Dxy, *Dxz, *Dyz, *vessTmp;
    size_t bytes = n * sizeof(float);
    cudaCheck(cudaMalloc(&tmp1, bytes));
    cudaCheck(cudaMalloc(&tmp2, bytes));
    cudaCheck(cudaMalloc(&Dxx,  bytes));
    cudaCheck(cudaMalloc(&Dyy,  bytes));
    cudaCheck(cudaMalloc(&Dzz,  bytes));
    cudaCheck(cudaMalloc(&Dxy,  bytes));
    cudaCheck(cudaMalloc(&Dxz,  bytes));
    cudaCheck(cudaMalloc(&Dyz,  bytes));
    cudaCheck(cudaMalloc(&vessTmp, bytes));

    std::vector<double> sigmaList;
    for (double s = sigmaFrom; s <= sigmaTo + 1e-6; s += sigmaStep)
        sigmaList.push_back(s);

    int nBlocks = int((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    for (double sigma : sigmaList) {
        std::vector<double> gauss, deriv;
        buildGaussianAndFirstDerivativeKernels(gauss, deriv, sigma);
        int kernelLen = int(gauss.size());

        // Smoothing and derivatives
        launchSeparableConvolution(0, inputDev, tmp1, gauss, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(1, tmp1, tmp2, gauss, kernelLen, nRows, nCols, nSlices);
        launchSeparableConvolution(2, tmp2, tmp1, gauss, kernelLen, nRows, nCols, nSlices);

        launchSecondDerivatives(tmp1, Dxx, Dyy, Dzz, nRows, nCols, nSlices);

        launchCrossDerivatives(inputDev, Dxy, Dxz, Dyz, deriv, gauss, kernelLen, nRows, nCols, nSlices);

        float sigmaSq = float(sigma * sigma);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dxx, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dyy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dzz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dxy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dxz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, THREADS_PER_BLOCK>>>(Dyz, n, sigmaSq);
        cudaCheck(cudaGetLastError());

        double gamma = structureSensitivity * sigma;
        vesselnessFrangiKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
            Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, vessTmp, n, alpha, beta, gamma, bright);
        cudaCheck(cudaGetLastError());

        if (sigmaList.size() > 1)
            elementwiseMaxKernel<<<nBlocks, THREADS_PER_BLOCK>>>(vessTmp, outputDev, n);
        else
            cudaCheck(cudaMemcpy(outputDev, vessTmp, bytes, cudaMemcpyDeviceToDevice));
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(output);
    cudaFree(tmp1); cudaFree(tmp2); cudaFree(Dxx); cudaFree(Dyy); cudaFree(Dzz);
    cudaFree(Dxy); cudaFree(Dxz); cudaFree(Dyz); cudaFree(vessTmp);
    mxGPUDestroyGPUArray(input); mxGPUDestroyGPUArray(output);
}
