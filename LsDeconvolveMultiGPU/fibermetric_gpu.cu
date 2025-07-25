#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr double SQRT_TWO_PI = 2.5066282746310002; // sqrt(2π)
constexpr float TWO_PI_OVER_THREE = float(2.0 * M_PI / 3.0);
constexpr float GAUSS_HALFWIDTH_MULT = 8.0f;
constexpr float EPSILON = 1e-7f;
constexpr int THREADS_PER_BLOCK = 512;
constexpr double FINITE_DIFF_5PT_DIVISOR = 12.0;

#define cudaCheck(call) \
    do { cudaError_t err = (call); if (err != cudaSuccess) \
    mexErrMsgIdAndTxt("fibermetric_gpu:cuda", \
    "CUDA error %s (%d) @ %s:%d", cudaGetErrorString(err), err, __FILE__, __LINE__); } while (0)

__device__ __host__ __forceinline__
size_t linearIndex3D(int row, int col, int slice, int nRows, int nCols) noexcept {
    // Use size_t to prevent integer overflow for large volumes
    return static_cast<size_t>(row)
         + static_cast<size_t>(col)   * static_cast<size_t>(nRows)
         + static_cast<size_t>(slice) * static_cast<size_t>(nRows) * nCols;
}

//---------------- Device-side Gaussian/Derivative Kernel (double for accuracy) -------------
__global__ void generateGaussianAndDerivativeKernels(
    float* __restrict__ gaussKernel, float* __restrict__ derivKernel,
    int kernelLen, double sigma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kernelLen) return;
    int half = kernelLen / 2;
    int x = i - half;
    double sigma2 = sigma * sigma;
    double norm = 1.0 / (SQRT_TWO_PI * sigma);
    double g = norm * exp(-0.5 * (x * x) / sigma2);
    gaussKernel[i] = static_cast<float>(g);
    derivKernel[i] = static_cast<float>(-x * g / sigma2);
}

void buildGaussianAndDerivativeKernelsDevice(
    float*& gaussKernelDev, float*& derivKernelDev,
    int kernelLen, double sigma, int threadsPerBlock)
{
    size_t kernelBytes = size_t(kernelLen) * sizeof(float);
    cudaCheck(cudaMalloc(&gaussKernelDev, kernelBytes));
    cudaCheck(cudaMalloc(&derivKernelDev, kernelBytes));
    int blocks = (kernelLen + threadsPerBlock - 1) / threadsPerBlock;
    generateGaussianAndDerivativeKernels<<<blocks, threadsPerBlock>>>(gaussKernelDev, derivKernelDev, kernelLen, sigma);
    cudaCheck(cudaGetLastError());
}

//---------------- Separable 1D Convolution (fmaf, in-place option) -------------------------
template<int AXIS>
__global__ void separableConvolution1DDeviceKernelFlat(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ kernelDev, int kernelLen,
    int nRows, int nCols, int nSlices)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(nRows) * nCols * nSlices;
    if (idx >= total) return;

    int row   = int(idx % nRows);
    int col   = int((idx / nRows) % nCols);
    int slice = int(idx / (size_t(nRows) * nCols));

    int halfWidth = kernelLen >> 1;
    float acc = 0.f;
    for (int k = -halfWidth; k <= halfWidth; ++k) {
        int rr = row, cc = col, ss = slice;
        if (AXIS == 0) rr += k;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;
        // Reflective boundary handling
        rr = (rr < 0) ? -rr : (rr >= nRows   ? 2 * nRows   - rr - 2 : rr);
        cc = (cc < 0) ? -cc : (cc >= nCols   ? 2 * nCols   - cc - 2 : cc);
        ss = (ss < 0) ? -ss : (ss >= nSlices ? 2 * nSlices - ss - 2 : ss);
        acc = fmaf(kernelDev[k + halfWidth], input[linearIndex3D(rr, cc, ss, nRows, nCols)], acc);
    }
    output[idx] = acc;
}

void launchSeparableConvolutionDevice(
    int axis, const float* inputDev, float* outputDev, const float* kernelDev,
    int kernelLen, int nRows, int nCols, int nSlices, int threadsPerBlock)
{
    size_t total = size_t(nRows) * nCols * nSlices;
    size_t nBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    if (nBlocks > 0x7FFFFFFF)
        mexErrMsgIdAndTxt("fibermetric_gpu:launch", "Too many CUDA blocks requested (%llu)", (unsigned long long)nBlocks);

    if (axis == 0)
        separableConvolution1DDeviceKernelFlat<0><<<nBlocks, threadsPerBlock>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else if (axis == 1)
        separableConvolution1DDeviceKernelFlat<1><<<nBlocks, threadsPerBlock>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else
        separableConvolution1DDeviceKernelFlat<2><<<nBlocks, threadsPerBlock>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);

    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
}

//-------------------- 5-point Second Derivative (fmaf, precompute squares) ----------------------
template<int AXIS>
__global__ void secondDerivative5ptKernel(
    const float* __restrict__ input, float* __restrict__ output,
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
        // Reflect at borders
        rr[k] = (rr[k] < 0) ? -rr[k] : (rr[k] >= nRows   ? 2 * nRows   - rr[k] - 2 : rr[k]);
        cc[k] = (cc[k] < 0) ? -cc[k] : (cc[k] >= nCols   ? 2 * nCols   - cc[k] - 2 : cc[k]);
        ss[k] = (ss[k] < 0) ? -ss[k] : (ss[k] >= nSlices ? 2 * nSlices - ss[k] - 2 : ss[k]);
    }
    // Store input value for each neighbor only once
    float in0 = input[linearIndex3D(rr[0], cc[0], ss[0], nRows, nCols)];
    float in1 = input[linearIndex3D(rr[1], cc[1], ss[1], nRows, nCols)];
    float in2 = input[linearIndex3D(rr[2], cc[2], ss[2], nRows, nCols)];
    float in3 = input[linearIndex3D(rr[3], cc[3], ss[3], nRows, nCols)];
    float in4 = input[linearIndex3D(rr[4], cc[4], ss[4], nRows, nCols)];
    float v = fmaf( -1.0f, in0, 0.f);
    v = fmaf( 16.0f, in1, v);
    v = fmaf(-30.0f, in2, v);
    v = fmaf( 16.0f, in3, v);
    v = fmaf( -1.0f, in4, v);
    output[linearIndex3D(row, col, slice, nRows, nCols)] = v / float(FINITE_DIFF_5PT_DIVISOR);
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

//---------------- Cross Derivatives: chain of separable convs (reuse buffers) ----------------
void launchCrossDerivativesDevice(
    const float* inputDev, float* Dxy, float* Dxz, float* Dyz,
    float* tmp1, float* tmp2,
    const float* derivKernelDev, const float* gaussKernelDev, int kernelLen,
    int nRows, int nCols, int nSlices, int threadsPerBlock)
{
    // Dxy: d2/dxdy
    launchSeparableConvolutionDevice(0, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(1, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(2, tmp2, Dxy, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    // Dxz: d2/dxdz
    launchSeparableConvolutionDevice(0, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(2, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(1, tmp2, Dxz, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    // Dyz: d2/dydz
    launchSeparableConvolutionDevice(1, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(2, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
    launchSeparableConvolutionDevice(0, tmp2, Dyz, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
}

//-------------------- Eigenvalue Decomposition (fmaf, float only) ----------------------
template <typename T>
__device__ __host__ __forceinline__ void swapCUDA(T& a, T& b) noexcept { T tmp = a; a = b; b = tmp; }

__device__ __host__ __forceinline__
void computeSymmetricEigenvalues3x3(
    float A11, float A22, float A33, float A12, float A13, float A23,
    float& l1, float& l2, float& l3) noexcept
{
    // All computations in float, but always use fmaf and avoid unnecessary sqrt
    float q = (A11 + A22 + A33) / 3.f;
    float B11 = A11 - q, B22 = A22 - q, B33 = A33 - q;
    float A12Sq = A12*A12, A13Sq = A13*A13, A23Sq = A23*A23;
    float p2 = fmaf(B11, B11, fmaf(B22, B22, fmaf(B33, B33, 2.f*(fmaf(A12Sq, 1.f, fmaf(A13Sq, 1.f, A23Sq)))))) / 6.f;
    float p = sqrtf(p2 + EPSILON);
    if (p < 1e-8f) { l1 = l2 = l3 = q; return; }
    float C11 = B11 / p, C22 = B22 / p, C33 = B33 / p;
    float C12 = A12 / p, C13 = A13 / p, C23 = A23 / p;
    float detC =
        C11 * (C22 * C33 - C23 * C23)
      - C12 * (C12 * C33 - C13 * C23)
      + C13 * (C12 * C23 - C13 * C22);
    float r = fmaxf(fminf(detC * 0.5f, 1.f), -1.f);
    float phi = acosf(r) / 3.f;
    float cosPhi = cosf(phi);
    float cosPhiShift = cosf(TWO_PI_OVER_THREE + phi);
    float twiceP = 2.f * p;
    float x1 = fmaf(twiceP, cosPhi, q);
    float x3 = fmaf(twiceP, cosPhiShift, q);
    float x2 = fmaf(3.f, q, -x1 - x3);

    float vals[3] = {x1, x2, x3};
    float absVals[3] = {fabsf(x1), fabsf(x2), fabsf(x3)};
    int order[3] = {0, 1, 2};
    if (absVals[0] > absVals[1]) { swapCUDA(order[0], order[1]); swapCUDA(absVals[0], absVals[1]); }
    if (absVals[1] > absVals[2]) { swapCUDA(order[1], order[2]); swapCUDA(absVals[1], absVals[2]); }
    if (absVals[0] > absVals[1]) { swapCUDA(order[0], order[1]); swapCUDA(absVals[0], absVals[1]); }
    l1 = vals[order[0]]; l2 = vals[order[1]]; l3 = vals[order[2]];
}

// --- Eigenvalue Kernel: computes and stores l1,l2,l3 for each voxel
__global__ void hessianToEigenvaluesKernel(
    const float* __restrict__ Dxx, const float* __restrict__ Dyy, const float* __restrict__ Dzz,
    const float* __restrict__ Dxy, const float* __restrict__ Dxz, const float* __restrict__ Dyz,
    float* __restrict__ l1, float* __restrict__ l2, float* __restrict__ l3, size_t n)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    computeSymmetricEigenvalues3x3(
        Dxx[idx], Dyy[idx], Dzz[idx], Dxy[idx], Dxz[idx], Dyz[idx], l1[idx], l2[idx], l3[idx]);
}

//-------------------- Vesselness (Frangi/Ferengi & Sato) kernels, float only -------------
__global__ void vesselnessFrangiKernelFromEigen(
    const float* __restrict__ l1, const float* __restrict__ l2, const float* __restrict__ l3,
    float* __restrict__ vesselness, size_t n,
    const float inv2Alpha2, const float inv2Beta2, const float inv2Gamma2,
    bool bright, bool doScaleNorm, float scaleNorm)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val1 = l1[idx], val2 = l2[idx], val3 = l3[idx];
    bool condition = bright ? (val2 < 0.f && val3 < 0.f) : (val2 > 0.f && val3 > 0.f);
    float response = 0.f;
    if (condition) {
        float absL1 = fabsf(val1), absL2 = fabsf(val2), absL3 = fabsf(val3);
        float Ra = absL2 / (absL3 + EPSILON);
        float Rb = absL1 / (sqrtf(fmaf(absL2, absL3, EPSILON)));
        float S2 = fmaf(val1, val1, fmaf(val2, val2, val3 * val3));
        float expRa = __expf(-Ra * Ra * inv2Alpha2);
        float expRb = __expf(-Rb * Rb * inv2Beta2);
        float expS2 = __expf(-S2      * inv2Gamma2);
        float tmp = fmaf(-expRa, expRb, expRb);
        response = fmaf(-expS2, tmp, tmp);
        if (doScaleNorm) response *= scaleNorm;
    }
    vesselness[idx] = response;
}

__global__ void vesselnessSatoKernelFromEigen(
    const float* __restrict__ l1,    // Eigenvalue 1 array
    const float* __restrict__ l2,    // Eigenvalue 2 array
    const float* __restrict__ l3,    // Eigenvalue 3 array
    float* __restrict__ vesselness,  // Output vesselness
    size_t n,                        // Number of voxels
    const float inv2Alpha2,          // 1/(2*alpha^2) (precomputed in double, cast to float)
    const float inv2Beta2,           // 1/(2*beta^2)
    bool bright                      // Polarity flag
)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ev1 = l1[idx], ev2 = l2[idx], ev3 = l3[idx];

    bool isVessel = bright ? (ev2 < 0.f && ev3 < 0.f)
                           : (ev2 > 0.f && ev3 > 0.f);

    float response = 0.f;
    if (isVessel) {
        float absEv1 = fabsf(ev1);
        float absEv2 = fabsf(ev2);
        float absEv3 = fabsf(ev3);

        float expL1 = __expf(-absEv1 * absEv1 * inv2Alpha2);
        float expL3 = __expf(-absEv3 * absEv3 * inv2Beta2);

        float absEv2TimesExpL1 = absEv2 * expL1;
        response = fmaf(-expL3, absEv2TimesExpL1, absEv2TimesExpL1);
    }

    vesselness[idx] = response;
}

__global__ void elementwiseMaxKernel(const float* src, float* dst, size_t n)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaxf(dst[idx], src[idx]);
}

__global__ void scaleArrayInPlaceKernel(float* arr, size_t n, float factor)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] *= factor;
}

//-------------------- Main MEX Entry (minimal VRAM, buffer reuse, cleanup early) ------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 9)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Usage: fibermetric_gpu(gpuArraySingle3D, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, structureSensitivity, 'bright'|'dark', 'frangi'|'sato')");

    mxInitGPU();
    int deviceId;
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess)
        mexErrMsgIdAndTxt("fibermetric_gpu:cuda", "cudaGetDevice failed: %s", cudaGetErrorString(err));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
    int threadsPerBlock = std::min(THREADS_PER_BLOCK, prop.maxThreadsPerBlock);

    //--- Inputs ---
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
    float alpha      = float(mxGetScalar(prhs[4]));
    float beta       = float(mxGetScalar(prhs[5]));
    float structureSensitivity = float(mxGetScalar(prhs[6]));
    char polarityBuf[16]; mxGetString(prhs[7], polarityBuf, sizeof(polarityBuf));
    bool bright = (strcmp(polarityBuf, "bright") == 0);
    char methodBuf[16]; mxGetString(prhs[8], methodBuf, sizeof(methodBuf));
    bool useFrangi = (strcmp(methodBuf, "frangi") == 0);
    bool useSato   = (strcmp(methodBuf, "sato") == 0);
    if (!useFrangi && !useSato)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage", "Last argument must be 'frangi' or 'sato'.");

    //--- Output Allocation ---
    mxGPUArray* output = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(output));
    cudaCheck(cudaMemset(outputDev, 0, n * sizeof(float)));

    //--- Workspace Buffers (only 2 temp) ---
    float *tmp1, *tmp2;
    size_t bytes = n * sizeof(float);
    cudaCheck(cudaMalloc(&tmp1,   bytes));
    cudaCheck(cudaMalloc(&tmp2,   bytes));

    //--- Sigma List and Scale Norm ---
    std::vector<double> sigmaList;
    for (double s = sigmaFrom; s <= sigmaTo + 1e-7; s += sigmaStep)
        sigmaList.push_back(s);
    bool singleSigma = (sigmaList.size() == 1);
    float scaleNorm = 1.f;
    bool doScaleNorm = false;
    if (singleSigma && std::fabs(sigmaList[0] - 1.0) < 1e-3) {
        doScaleNorm = true;
        scaleNorm = 0.015f;
    }
    int nBlocks = int((n + threadsPerBlock - 1) / threadsPerBlock);

    // --- Allocate eigenvalue arrays (always n*3) ---
    float *l1, *l2, *l3;
    cudaCheck(cudaMalloc(&l1, bytes));
    cudaCheck(cudaMalloc(&l2, bytes));
    cudaCheck(cudaMalloc(&l3, bytes));

    //--- Main Loop Over Scales ---
    for (double sigma : sigmaList)
    {
        int halfWidth = int(std::ceil(GAUSS_HALFWIDTH_MULT * sigma));
        int kernelLen = 2 * halfWidth + 1;
        float* gaussKernelDev = nullptr;
        float* derivKernelDev = nullptr;
        buildGaussianAndDerivativeKernelsDevice(gaussKernelDev, derivKernelDev, kernelLen, sigma, threadsPerBlock);

        // Gaussian smoothing (tmp1 <-> tmp2, ends in tmp1)
        launchSeparableConvolutionDevice(0, inputDev, tmp1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(1, tmp1, tmp2, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(2, tmp2, tmp1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);

        // --- Allocate Hessian/cross-derivative buffers (min lifetime!) ---
        float *Dxx, *Dyy, *Dzz, *Dxy, *Dxz, *Dyz;
        cudaCheck(cudaMalloc(&Dxx, bytes));
        cudaCheck(cudaMalloc(&Dyy, bytes));
        cudaCheck(cudaMalloc(&Dzz, bytes));
        cudaCheck(cudaMalloc(&Dxy, bytes));
        cudaCheck(cudaMalloc(&Dxz, bytes));
        cudaCheck(cudaMalloc(&Dyz, bytes));

        // Hessian diagonals
        launchSecondDerivatives(tmp1, Dxx, Dyy, Dzz, nRows, nCols, nSlices);
        cudaCheck(cudaGetLastError());

        // Cross-derivatives
        launchCrossDerivativesDevice(inputDev, Dxy, Dxz, Dyz, tmp1, tmp2,
            derivKernelDev, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
        cudaCheck(cudaGetLastError());

        cudaCheck(cudaDeviceSynchronize());
        cudaFree(gaussKernelDev);
        cudaFree(derivKernelDev);

        // Scale normalization
        float sigmaSq = float(sigma * sigma);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dxx, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dyy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dzz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dxy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dxz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock>>>(Dyz, n, sigmaSq);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // --- Compute eigenvalues: l1, l2, l3, then free Hessians immediately! ---
        hessianToEigenvaluesKernel<<<nBlocks, threadsPerBlock>>>(
            Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, l1, l2, l3, n);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaFree(Dxx); cudaFree(Dyy); cudaFree(Dzz);
        cudaFree(Dxy); cudaFree(Dxz); cudaFree(Dyz);

        // --- Vesselness kernel on eigenvalues only ---
        float* vessTmp = tmp1; // reuse tmp1 for vesselness result!

        float gamma = structureSensitivity * float(sigma);
        const float inv2Alpha2 = static_cast<float>(1.0 / (2.0 * double(alpha) * double(alpha)));
        const float inv2Beta2  = static_cast<float>(1.0 / (2.0 * double(beta)  * double(beta )));
        const float inv2Gamma2 = static_cast<float>(1.0 / (2.0 * double(gamma) * double(gamma)));
        if (useFrangi) {
            vesselnessFrangiKernelFromEigen<<<nBlocks, threadsPerBlock>>>(
                l1, l2, l3, vessTmp, n, inv2Alpha2, inv2Beta2, inv2Gamma2, bright, doScaleNorm, scaleNorm);
        } else {
            vesselnessSatoKernelFromEigen<<<nBlocks, threadsPerBlock>>>(
                l1, l2, l3, vessTmp, n, inv2Alpha2, inv2Beta2, bright);
        }
        cudaCheck(cudaGetLastError());

        // --- Multi-sigma: max-projection; single: copy
        if (!singleSigma)
            elementwiseMaxKernel<<<nBlocks, threadsPerBlock>>>(vessTmp, outputDev, n);
        else
            cudaCheck(cudaMemcpy(outputDev, vessTmp, bytes, cudaMemcpyDeviceToDevice));
    }

    //--- Cleanup ---
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);
    cudaCheck(cudaDeviceSynchronize());
    cudaFree(tmp1); cudaFree(tmp2);
    cudaFree(l1); cudaFree(l2); cudaFree(l3);
    mxGPUDestroyGPUArray(input); mxGPUDestroyGPUArray(output);
}
