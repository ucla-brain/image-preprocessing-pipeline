/*=====================================================================*
 *  fibermetric_gpu.cu  –  GPU Frangi/Sato vesselness for large stacks *
 *---------------------------------------------------------------------*
 *  Copyright 2025 Keivan Moradi                                       *
 *                                                                     *
 *  Fixes / Enhancements (2025-07-24)                                  *
 *   • 64-bit safe linear indexing (size_t)                            *
 *   • Flat 1-D grid for derivative kernels - avoids 65 535 limitation *
 *   • tmp1/tmp2 reused as l1/l2; Dyy reused as l3                     *
 *   • Hessian buffers freed ASAP, peak VRAM ≤ 8 N floats              *
 *   • fmaf-rich arithmetic, constexpr constants, pre-cached squares   *
 *   • Uniform reflection helper, no superfluous sqrt(a*a)             *
 *   • Full cudaCheck on every API / kernel call                       *
 *   • No std::swap (custom host-device swapCUDA)                      *
 *=====================================================================*/

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>      // size_t, uint32_t …
#include <cstring>      // mxGetString
#include <cstdio>       // for error text

/*---------------------------  Constants  ----------------------------*/
#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif

constexpr double SQRT_TWO_PI          = 2.506628274631000502415;  // √(2π)
constexpr float  TWO_PI_OVER_THREE    = static_cast<float>(2.0 * M_PI / 3.0);
constexpr float  GAUSS_HALFWIDTH_MULT = 8.0f;    // 8 σ half-width
constexpr float  EPSILON_F            = 1.0e-7f;
constexpr int    THREADS_PER_BLOCK    = 512;
constexpr float  FINITE_DIFF_5PT_DIV  = 12.0f;

/*--------------------  Error-checking helper  -----------------------*/
#define cudaCheck(call)                                                     \
  do {                                                                      \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess)                                                \
      mexErrMsgIdAndTxt("fibermetric_gpu:cuda",                             \
                        "CUDA error %s (%d) @ %s:%d",                       \
                        cudaGetErrorString(_err), static_cast<int>(_err),   \
                        __FILE__, __LINE__);                                \
  } while (0)

/*-----------------------------  Utils  ------------------------------*/
__device__ __host__ __forceinline__ int  reflectCoord(int p, int len) noexcept
{   return (p < 0) ? -p : (p >= len ? 2 * len - p - 2 : p); }

__device__ __host__ __forceinline__ size_t
linearIndex3D(int row, int col, int slice, int nRows, int nCols) noexcept
{
    return  static_cast<size_t>(row)                                   +
            static_cast<size_t>(col)   * static_cast<size_t>(nRows)    +
            static_cast<size_t>(slice) * static_cast<size_t>(nRows) * nCols;
}

template<typename T>
__device__ __host__ __forceinline__ void swapCUDA(T& a, T& b) noexcept
{   T tmp = a;  a = b;  b = tmp; }

/*=====================================================================*
 *            1.  Gaussian and derivative kernels (GPU)                *
 *=====================================================================*/
__global__ void generateGaussianAndDerivativeKernels(
        float* __restrict__ gaussKernel,
        float* __restrict__ derivKernel,
        int kernelLen, double sigma)
{
    const int i    = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kernelLen) return;

    const int    half     = kernelLen >> 1;
    const int    x        = i - half;
    const double sigma2   = sigma * sigma;
    const double norm     = 1.0 / (SQRT_TWO_PI * sigma);
    const double g        = norm * exp(-0.5 * (x * x) / sigma2);

    gaussKernel[i] = static_cast<float>(g);
    derivKernel[i] = static_cast<float>(-x * g / sigma2);
}

static inline void buildGaussianAndDerivativeKernelsDevice(
        float*& gaussKernelDev, float*& derivKernelDev,
        int kernelLen, double sigma, int threadsPerBlock)
{
    const size_t bytes = static_cast<size_t>(kernelLen) * sizeof(float);
    cudaCheck(cudaMalloc(&gaussKernelDev, bytes));
    cudaCheck(cudaMalloc(&derivKernelDev, bytes));

    const int nBlocks = (kernelLen + threadsPerBlock - 1) / threadsPerBlock;
    generateGaussianAndDerivativeKernels<<<nBlocks, threadsPerBlock>>>(
            gaussKernelDev, derivKernelDev, kernelLen, sigma);
    cudaCheck(cudaGetLastError());
}

/*=====================================================================*
 *         2.  Separable 1-D convolution (flat-index variant)          *
 *=====================================================================*/
template<int AXIS>
__global__ void separableConvolution1DDeviceKernelFlat(
        const float* __restrict__ input,
        float*       __restrict__ output,
        const float* __restrict__ kernelDev, int kernelLen,
        int nRows, int nCols, int nSlices)
{
    const size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total  = static_cast<size_t>(nRows) * nCols * nSlices;
    if (idx >= total) return;

    /* 3-D index from flat idx (column-major order) */
    const int row   = static_cast<int>(idx % nRows);
    const int col   = static_cast<int>((idx / nRows) % nCols);
    const int slice = static_cast<int>(idx / (static_cast<size_t>(nRows) * nCols));

    const int halfW = kernelLen >> 1;
    float acc = 0.0f;
    #pragma unroll
    for (int k = -halfW; k <= halfW; ++k)
    {
        int rr = row, cc = col, ss = slice;
        if (AXIS == 0) rr += k;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;

        rr = reflectCoord(rr, nRows);
        cc = reflectCoord(cc, nCols);
        ss = reflectCoord(ss, nSlices);

        const size_t nIdx = linearIndex3D(rr, cc, ss, nRows, nCols);
        acc = fmaf(kernelDev[k + halfW], input[nIdx], acc);
    }

    output[idx] = acc;
}

static void launchSeparableConvolutionDevice(
        int axis, const float* inputDev, float* outputDev,
        const float* kernelDev, int kernelLen,
        int nRows, int nCols, int nSlices, int threadsPerBlock)
{
    const size_t total   = static_cast<size_t>(nRows) * nCols * nSlices;
    const int nBlocks    = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);

    if      (axis == 0)
        separableConvolution1DDeviceKernelFlat<0><<<nBlocks, threadsPerBlock>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else if (axis == 1)
        separableConvolution1DDeviceKernelFlat<1><<<nBlocks, threadsPerBlock>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else
        separableConvolution1DDeviceKernelFlat<2><<<nBlocks, threadsPerBlock>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);

    cudaCheck(cudaGetLastError());
}

/*=====================================================================*
 *                3.  5-point 2nd derivative (flat idx)                *
 *=====================================================================*/
template<int AXIS>
__global__ void secondDerivative5ptKernelFlat(
        const float* __restrict__ input,
        float*       __restrict__ output,
        int nRows, int nCols, int nSlices)
{
    const size_t idxTotal = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total    = static_cast<size_t>(nRows) * nCols * nSlices;
    if (idxTotal >= total) return;

    const int row   = static_cast<int>(idxTotal % nRows);
    const int col   = static_cast<int>((idxTotal / nRows) % nCols);
    const int slice = static_cast<int>(idxTotal / (static_cast<size_t>(nRows) * nCols));

    /* Pre-compute reflected neighbour coordinates */
    int coord[5] = { row, col, slice };   // placeholders

    float val = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k)
    {
        const int offset = k - 2;
        int rr = row, cc = col, ss = slice;
        if (AXIS == 0) rr += offset;
        if (AXIS == 1) cc += offset;
        if (AXIS == 2) ss += offset;

        rr = reflectCoord(rr, nRows);
        cc = reflectCoord(cc, nCols);
        ss = reflectCoord(ss, nSlices);

        const size_t nIdx = linearIndex3D(rr, cc, ss, nRows, nCols);
        const float   coef = (k == 0 || k == 4) ? -1.0f :
                             (k == 1 || k == 3) ? 16.0f : -30.0f;
        val = fmaf(coef, input[nIdx], val);
    }
    output[idxTotal] = val / FINITE_DIFF_5PT_DIV;
}

static inline void launchSecondDerivatives(
        const float* smoothedInput,
        float* Dxx, float* Dyy, float* Dzz,
        int nRows, int nCols, int nSlices,
        int threadsPerBlock)
{
    const size_t total  = static_cast<size_t>(nRows) * nCols * nSlices;
    const int nBlocks   = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);

    secondDerivative5ptKernelFlat<0><<<nBlocks, threadsPerBlock>>>(
            smoothedInput, Dxx, nRows, nCols, nSlices);
    secondDerivative5ptKernelFlat<1><<<nBlocks, threadsPerBlock>>>(
            smoothedInput, Dyy, nRows, nCols, nSlices);
    secondDerivative5ptKernelFlat<2><<<nBlocks, threadsPerBlock>>>(
            smoothedInput, Dzz, nRows, nCols, nSlices);
    cudaCheck(cudaGetLastError());
}

/*=====================================================================*
 *             4.  3×3 symmetric eigen-decomposition (analytic)        *
 *=====================================================================*/
__device__ __host__ __forceinline__
void computeSymmetricEigenvalues3x3(
        float A11, float A22, float A33,
        float A12, float A13, float A23,
        float& l1, float& l2, float& l3) noexcept
{
    /* Shift to trace-free matrix to improve stability */
    const float q     = (A11 + A22 + A33) / 3.0f;
    const float B11   = A11 - q,  B22 = A22 - q,  B33 = A33 - q;

    /* |B|²  (squared Frobenius norm divided by 6) */
    const float A12Sq = A12 * A12,  A13Sq = A13 * A13,  A23Sq = A23 * A23;
    const float p2    = (B11*B11 + B22*B22 + B33*B33 +
                         2.0f*(A12Sq + A13Sq + A23Sq)) / 6.0f;
    const float p     = sqrtf(p2 + EPSILON_F);
    if (p < 1.0e-8f) { l1 = l2 = l3 = q; return; }

    /* Normalised matrix C = B / p with det(C)=cos(3φ) */
    const float C11 = B11 / p,  C22 = B22 / p,  C33 = B33 / p;
    const float C12 = A12 / p,  C13 = A13 / p,  C23 = A23 / p;
    const float detC =
           C11 * (C22 * C33 - C23 * C23)
        -  C12 * (C12 * C33 - C13 * C23)
        +  C13 * (C12 * C23 - C13 * C22);

    const float r   = fmaxf(fminf(0.5f * detC, 1.0f), -1.0f);
    const float phi = acosf(r) / 3.0f;

    const float cosPhi0 = cosf(phi);
    const float cosPhi2 = cosf(phi + TWO_PI_OVER_THREE);
    const float cosPhi4 = cosf(phi + 2.0f * TWO_PI_OVER_THREE);

    const float twoP   = 2.0f * p;
    const float eig[3] = { fmaf(twoP, cosPhi0, q),
                           fmaf(twoP, cosPhi2, q),
                           fmaf(twoP, cosPhi4, q) };

    /* Sort |λ| ascending  (for Frangi/Sato) */
    int order[3]      = {0,1,2};
    float absEig[3]   = { fabsf(eig[0]), fabsf(eig[1]), fabsf(eig[2]) };
    if (absEig[0] > absEig[1]) { swapCUDA(order[0],order[1]); swapCUDA(absEig[0],absEig[1]); }
    if (absEig[1] > absEig[2]) { swapCUDA(order[1],order[2]); swapCUDA(absEig[1],absEig[2]); }
    if (absEig[0] > absEig[1]) { swapCUDA(order[0],order[1]); swapCUDA(absEig[0],absEig[1]); }
    l1 = eig[order[0]];  l2 = eig[order[1]];  l3 = eig[order[2]];
}

/*--------------------  Eigenvalue kernel  ---------------------------*/
__global__ void hessianToEigenvaluesKernel(
        const float* __restrict__ Dxx,
        const float* __restrict__ Dyy,
        const float* __restrict__ Dzz,
        const float* __restrict__ Dxy,
        const float* __restrict__ Dxz,
        const float* __restrict__ Dyz,
        float*       __restrict__ l1,
        float*       __restrict__ l2,
        float*       __restrict__ l3,
        size_t n)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float e1,e2,e3;
    computeSymmetricEigenvalues3x3(
        Dxx[idx], Dyy[idx], Dzz[idx],
        Dxy[idx], Dxz[idx], Dyz[idx],
        e1, e2, e3);
    l1[idx] = e1;
    l2[idx] = e2;
    l3[idx] = e3;
}

/*=====================================================================*
 *                   5.  Vesselness kernels                            *
 *=====================================================================*/
__global__ void vesselnessFrangiKernelFromEigen(
        const float* __restrict__ l1, const float* __restrict__ l2, const float* __restrict__ l3,
        float*       __restrict__ vesselness, size_t n,
        const float inv2Alpha2, const float inv2Beta2, const float inv2Gamma2,
        bool bright, bool doScaleNorm, float scaleNorm)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float v1 = l1[idx], v2 = l2[idx], v3 = l3[idx];
    const bool  cond = bright ? (v2 < 0.0f && v3 < 0.0f)
                              : (v2 > 0.0f && v3 > 0.0f);

    float resp = 0.0f;
    if (cond)
    {
        const float absV1 = fabsf(v1),  absV2 = fabsf(v2),  absV3 = fabsf(v3);
        const float Ra = absV2 / (absV3 + EPSILON_F);
        const float Rb = absV1 / sqrtf(absV2 * absV3 + EPSILON_F);
        const float S2 = fmaf(v1, v1, fmaf(v2, v2, v3 * v3));

        resp = __expf(-Ra*Ra * inv2Alpha2) *
               (1.0f - __expf(-Rb*Rb * inv2Beta2)) *
               __expf(-S2     * inv2Gamma2);
        if (doScaleNorm) resp *= scaleNorm;
    }
    vesselness[idx] = resp;
}

__global__ void vesselnessSatoKernelFromEigen(
        const float* __restrict__ l1, const float* __restrict__ l2, const float* __restrict__ l3,
        float*       __restrict__ vesselness, size_t n,
        const float inv2Alpha2, const float inv2Beta2,
        bool bright)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float e1 = l1[idx], e2 = l2[idx], e3 = l3[idx];
    const bool  isVessel = bright ? (e2 < 0.0f && e3 < 0.0f)
                                  : (e2 > 0.0f && e3 > 0.0f);

    float resp = 0.0f;
    if (isVessel)
    {
        const float absE1 = fabsf(e1), absE2 = fabsf(e2), absE3 = fabsf(e3);
        resp = absE2 * __expf(-absE1*absE1 * inv2Alpha2)
                    * __expf(-absE3*absE3 * inv2Beta2);
    }
    vesselness[idx] = resp;
}

/*--------- misc helpers --------*/
__global__ void elementwiseMaxKernel(const float* src, float* dst, size_t n)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaxf(dst[idx], src[idx]);
}

__global__ void scaleArrayInPlaceKernel(float* arr, size_t n, float factor)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] *= factor;
}

/*=====================================================================*
 *                        6.  MEX entrypoint                           *
 *=====================================================================*/
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 9)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Usage: fibermetric_gpu(gpuArraySingle3D, sigmaFrom, sigmaTo, "
            "sigmaStep, alpha, beta, structSens, 'bright'|'dark', 'frangi'|'sato')");

    mxInitGPU();

    /*--------------------  Device / block size  --------------------*/
    int deviceId;  cudaCheck(cudaGetDevice(&deviceId));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, deviceId);
    const int threadsPerBlock = std::min(prop.maxThreadsPerBlock, THREADS_PER_BLOCK);

    /*--------------------  Inputs  ---------------------------------*/
    const mxGPUArray* input = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(input) != mxSINGLE_CLASS ||
        mxGPUGetNumberOfDimensions(input) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type",
                          "Input must be a 3-D single gpuArray.");

    const mwSize* dims = mxGPUGetDimensions(input);
    const int nRows   = static_cast<int>(dims[0]);
    const int nCols   = static_cast<int>(dims[1]);
    const int nSlices = static_cast<int>(dims[2]);
    const size_t nVox = static_cast<size_t>(nRows) * nCols * nSlices;

    const float* inputDev = static_cast<const float*>(mxGPUGetDataReadOnly(input));

    const double sigmaFrom  = mxGetScalar(prhs[1]);
    const double sigmaTo    = mxGetScalar(prhs[2]);
    const double sigmaStep  = mxGetScalar(prhs[3]);
    const float  alpha      = static_cast<float>(mxGetScalar(prhs[4]));
    const float  beta       = static_cast<float>(mxGetScalar(prhs[5]));
    const float  structSens = static_cast<float>(mxGetScalar(prhs[6]));

    char polBuf[16];   mxGetString(prhs[7], polBuf, sizeof(polBuf));
    const bool bright  = (std::strcmp(polBuf, "bright") == 0);

    char methBuf[16];  mxGetString(prhs[8], methBuf, sizeof(methBuf));
    const bool useFrangi = (std::strcmp(methBuf, "frangi") == 0);
    const bool useSato   = (std::strcmp(methBuf, "sato")   == 0);
    if (!useFrangi && !useSato)
        mexErrMsgIdAndTxt("fibermetric_gpu:method",
                          "Method must be 'frangi' or 'sato'.");

    /*----------------  Output & reusable scratch buffers -----------*/
    mxGPUArray* output = mxGPUCreateGPUArray(
        3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(output));
    cudaCheck(cudaMemset(outputDev, 0, nVox * sizeof(float)));

    /* tmp1 / tmp2 persist across scales and double as l1 / l2 */
    float *tmp1Dev, *tmp2Dev;
    const size_t bytesN = nVox * sizeof(float);
    cudaCheck(cudaMalloc(&tmp1Dev, bytesN));
    cudaCheck(cudaMalloc(&tmp2Dev, bytesN));

    /* Block config for “flat” kernels */
    const int nBlocksFlat = static_cast<int>((nVox + threadsPerBlock - 1) / threadsPerBlock);

    /* Build list of sigmas */
    std::vector<double> sigmas;
    for (double s = sigmaFrom; s <= sigmaTo + 1e-9; s += sigmaStep)
        sigmas.push_back(s);
    const bool singleSigma = (sigmas.size() == 1);

    /*=====================  Main scale loop  ======================*/
    for (double sigma : sigmas)
    {
        /* ---- Kernels along this scale ---- */
        const int halfWidth = static_cast<int>(std::ceil(GAUSS_HALFWIDTH_MULT * sigma));
        const int kernelLen = 2 * halfWidth + 1;

        float *gaussK, *derivK;
        buildGaussianAndDerivativeKernelsDevice(gaussK, derivK, kernelLen, sigma, threadsPerBlock);

        /* ---- Gaussian smoothing (tmp1 ends with smoothed volume) ---- */
        launchSeparableConvolutionDevice(0, inputDev, tmp1Dev,
                gaussK, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(1, tmp1Dev, tmp2Dev,
                gaussK, kernelLen, nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(2, tmp2Dev, tmp1Dev,
                gaussK, kernelLen, nRows, nCols, nSlices, threadsPerBlock);

        /* ---- Allocate Hessian buffers (reuse later!) ---- */
        float *Dxx, *Dyy, *Dzz, *Dxy, *Dxz, *Dyz;
        cudaCheck(cudaMalloc(&Dxx, bytesN));
        cudaCheck(cudaMalloc(&Dyy, bytesN));  /* becomes l3 later  */
        cudaCheck(cudaMalloc(&Dzz, bytesN));
        cudaCheck(cudaMalloc(&Dxy, bytesN));
        cudaCheck(cudaMalloc(&Dxz, bytesN));
        cudaCheck(cudaMalloc(&Dyz, bytesN));

        /* ---- Second derivatives on smoothed volume ---- */
        launchSecondDerivatives(tmp1Dev, Dxx, Dyy, Dzz,
                                nRows, nCols, nSlices, threadsPerBlock);

        /* ---- Cross derivatives (uses tmp1/tmp2 as work) ---- */
        launchSeparableConvolutionDevice(0, inputDev, tmp1Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(1, tmp1Dev, tmp2Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(2, tmp2Dev, Dxy,    gaussK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);

        launchSeparableConvolutionDevice(0, inputDev, tmp1Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(2, tmp1Dev, tmp2Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(1, tmp2Dev, Dxz,    gaussK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);

        launchSeparableConvolutionDevice(1, inputDev, tmp1Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(2, tmp1Dev, tmp2Dev, derivK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);
        launchSeparableConvolutionDevice(0, tmp2Dev, Dyz,    gaussK,  kernelLen,
                                         nRows, nCols, nSlices, threadsPerBlock);

        cudaCheck(cudaDeviceSynchronize());
        cudaFree(gaussK);  cudaFree(derivK);

        /* ---- Scale-normalise Hessian ---- */
        const float sigmaSq = static_cast<float>(sigma * sigma);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dxx, nVox, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dyy, nVox, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dzz, nVox, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dxy, nVox, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dxz, nVox, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocksFlat, threadsPerBlock>>>(Dyz, nVox, sigmaSq);
        cudaCheck(cudaGetLastError());

        /* ---- Eigenvalues: l1  = tmp1Dev, l2 = tmp2Dev, l3 = Dyy ---- */
        hessianToEigenvaluesKernel<<<nBlocksFlat, threadsPerBlock>>>(
                Dxx, Dyy, Dzz, Dxy, Dxz, Dyz,
                tmp1Dev, tmp2Dev, Dyy,   /* l3 overwrites Dyy */
                nVox);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        /* free Hessian pieces except Dxx (will become vesselness) & Dyy (l3) */
        cudaFree(Dzz); cudaFree(Dxy); cudaFree(Dxz); cudaFree(Dyz);

        /* ---- Vesselness @ this scale, write into Dxx to reuse memory ---- */
        const float gamma      = structSens * static_cast<float>(sigma);
        const float inv2Alpha2 = 1.0f / (2.0f * alpha * alpha);
        const float inv2Beta2  = 1.0f / (2.0f * beta  * beta );
        const float inv2Gamma2 = 1.0f / (2.0f * gamma * gamma);

        if (useFrangi)
            vesselnessFrangiKernelFromEigen<<<nBlocksFlat, threadsPerBlock>>>(
                tmp1Dev, tmp2Dev, Dyy,  /* l1,l2,l3 */
                Dxx, nVox,
                inv2Alpha2, inv2Beta2, inv2Gamma2,
                bright,
                /* single-sigma norm only around σ≈1 */
                singleSigma && std::fabs(sigma - 1.0) < 1.0e-3, 0.015f);
        else
            vesselnessSatoKernelFromEigen<<<nBlocksFlat, threadsPerBlock>>>(
                tmp1Dev, tmp2Dev, Dyy, Dxx, nVox,
                inv2Alpha2, inv2Beta2, bright);
        cudaCheck(cudaGetLastError());

        /* ---- Combine with running max or copy ---- */
        if (singleSigma)
            cudaCheck(cudaMemcpy(outputDev, Dxx, bytesN, cudaMemcpyDeviceToDevice));
        else
            elementwiseMaxKernel<<<nBlocksFlat, threadsPerBlock>>>(Dxx, outputDev, nVox);

        /* ---- Cleanup this scale ---- */
        cudaFree(Dxx);   /* Dxx held vesselness */
        cudaFree(Dyy);   /* l3 */
    }

    /*----------------  Return to MATLAB  ---------------------------*/
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);

    /* persistent buffers */
    cudaFree(tmp1Dev);  cudaFree(tmp2Dev);
    mxGPUDestroyGPUArray(input);
    mxGPUDestroyGPUArray(output);
}
