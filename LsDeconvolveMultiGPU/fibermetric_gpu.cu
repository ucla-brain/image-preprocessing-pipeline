#pragma once

// ==== MATLAB / CUDA headers =================================================
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==== STL / C headers =======================================================
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>      // strcmp
#include <limits>
#include <stdexcept>
#include <vector>

// ==== Constants =============================================================
#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif
constexpr float PI_OVER_THREE       = static_cast<float>(M_PI / 3.0);
constexpr float INV_SQRT_2PI        = 0.3989422804014327f;  // 1 / √(2π)
constexpr float GAUSSIAN_SIGMA_MULT = 3.0f;                 // kernel radius
constexpr float SQRT_TWO            = 1.4142135623730951f;  // √2

// ==== Utility macro =========================================================
#define cudaCheck(err)                                                     \
    do {                                                                   \
        cudaError_t _e = (err);                                            \
        if (_e != cudaSuccess)                                             \
            mexErrMsgIdAndTxt("fibermetric_gpu:cuda",                      \
                              "CUDA error %s at %s:%d",                    \
                              cudaGetErrorString(_e), __FILE__, __LINE__); \
    } while (0)

// ============================================================================
//                         1-D GAUSSIAN KERNELS
// ============================================================================
static inline void makeGaussianKernels1D(std::vector<float>& g,
                                         std::vector<float>& gx,
                                         int                 kSize,
                                         float               sigma)
{
    const int   halfK      = kSize / 2;
    const float sigma2     = sigma * sigma;
    const float invSigma2  = 1.0f / sigma2;
    const float invDenom   = INV_SQRT_2PI / sigma;
    float       sumG       = 0.0f;

    g.resize(kSize);
    gx.resize(kSize);

    for (int i = 0; i < kSize; ++i)
    {
        const int   x  = i - halfK;
        const float x2 = static_cast<float>(x * x);
        const float v  = expf(-0.5f * x2 * invSigma2) * invDenom;
        g [i] = v;
        gx[i] = -x * v * invSigma2;
        sumG += v;
    }
    for (float& v : g)  v /= sumG;      // keep Gaussian normalised
    for (float& v : gx) v *= SQRT_TWO;  // MATLAB’s √2 factor for derivatives
}

static inline void makeGaussianSecondDerivKernel1D(std::vector<float>& gxx, int kSize, float sigma)
{
    const int halfK = kSize / 2;
    const float sigma2 = sigma * sigma;
    const float sigma4 = sigma2 * sigma2;
    const float invSqrt2PiSigma = 1.0f / (sqrtf(2.0f * M_PI) * sigma);
    float sum = 0.0f;
    gxx.resize(kSize);

    for (int i = 0; i < kSize; ++i)
    {
        const int x = i - halfK;
        const float x2 = x * x;
        float val = (x2 - sigma2) / sigma4 * expf(-0.5f * x2 / sigma2) * invSqrt2PiSigma;
        gxx[i] = val;
        sum += x * x * fabs(val);
    }
    // No normalization needed for second derivatives
}


// ============================================================================
//                   SIMPLE 3-D ROW-MAJOR INDEXING (column major in MATLAB)
// ============================================================================
__device__ __host__ __forceinline__
int linearIndex3D(int row, int col, int slice,
                  int nRows, int nCols) noexcept
{
    return row + col * nRows + slice * nRows * nCols;
}

// ============================================================================
//                       SEPARABLE 1-D CONVOLUTIONS
// ============================================================================
template<int AXIS>
__global__ void convolve1D(const float* __restrict__ src,
                           float*       __restrict__ dst,
                           const float* __restrict__ k,
                           int kSize,
                           int nRows, int nCols, int nSlices)
{
    constexpr int HALF_K_DUMMY = 0; // suppress unused-var warning when templated
    const int halfK  = kSize / 2;
    const int row    = (AXIS == 0) ? blockIdx.x * blockDim.x + threadIdx.x
                                   : blockIdx.x;
    const int col    = (AXIS == 1) ? blockIdx.y * blockDim.y + threadIdx.y
                                   : blockIdx.y;
    const int slice  = (AXIS == 2) ? blockIdx.z * blockDim.z + threadIdx.z
                                   : blockIdx.z;

    if (row   >= nRows   || col >= nCols || slice >= nSlices) return;

    float sum = 0.0f;
#pragma unroll
    for (int kOff = -halfK; kOff <= halfK; ++kOff)
    {
        int r = row, c = col, s = slice;
        if      (AXIS == 0) r += kOff;
        else if (AXIS == 1) c += kOff;
        else                s += kOff;

        // symmetric boundary conditions
        if (r < 0)           r = -r;
        if (r >= nRows)      r = 2 * nRows   - r - 2;
        if (c < 0)           c = -c;
        if (c >= nCols)      c = 2 * nCols   - c - 2;
        if (s < 0)           s = -s;
        if (s >= nSlices)    s = 2 * nSlices - s - 2;

        sum = fmaf(src[linearIndex3D(r, c, s, nRows, nCols)], k[kOff + halfK], sum);
    }
    dst[linearIndex3D(row, col, slice, nRows, nCols)] = sum;
}

// launch helpers (keep host-side to avoid extra template instantiations)
static inline void launchConvX(const float* src, float* dst,
                               const float* k, int kSize,
                               int nRows, int nCols, int nSlices)
{
    dim3 block(32, 1, 1), grid((nRows + 31) / 32, nCols, nSlices);
    convolve1D<0><<<grid, block>>>(src, dst, k, kSize, nRows, nCols, nSlices);
}

static inline void launchConvY(const float* src, float* dst,
                               const float* k, int kSize,
                               int nRows, int nCols, int nSlices)
{
    dim3 block(1, 32, 1), grid(nRows, (nCols + 31) / 32, nSlices);
    convolve1D<1><<<grid, block>>>(src, dst, k, kSize, nRows, nCols, nSlices);
}

static inline void launchConvZ(const float* src, float* dst,
                               const float* k, int kSize,
                               int nRows, int nCols, int nSlices)
{
    dim3 block(1, 1, 32), grid(nRows, nCols, (nSlices + 31) / 32);
    convolve1D<2><<<grid, block>>>(src, dst, k, kSize, nRows, nCols, nSlices);
}

// ============================================================================
//              FINITE-DIFFERENCE SECOND-DERIVATIVE KERNELS
// ============================================================================
template<int AXIS>
__global__ void secondDerivative(const float* __restrict__ in,
                                 float*       __restrict__ out,
                                 int nRows, int nCols, int nSlices,
                                 float sigma2)
{
    const int row   = (AXIS == 0) ? blockIdx.x * blockDim.x + threadIdx.x
                                  : blockIdx.x;
    const int col   = (AXIS == 1) ? blockIdx.y * blockDim.y + threadIdx.y
                                  : blockIdx.y;
    const int slice = (AXIS == 2) ? blockIdx.z * blockDim.z + threadIdx.z
                                  : blockIdx.z;

    if (row >= nRows || col >= nCols || slice >= nSlices) return;

    const int idx = linearIndex3D(row, col, slice, nRows, nCols);

    int r1 = row, r2 = row;
    if      (AXIS == 0) { r1 = max(row   - 1, 0);  r2 = min(row   + 1, nRows   - 1); }
    else if (AXIS == 1) { r1 = max(col   - 1, 0);  r2 = min(col   + 1, nCols   - 1); }
    else                { r1 = max(slice - 1, 0);  r2 = min(slice + 1, nSlices - 1); }

    const float s_m =   (AXIS == 0) ? in[linearIndex3D(r1, col, slice, nRows, nCols)]
                      : (AXIS == 1) ? in[linearIndex3D(row, r1, slice, nRows, nCols)]
                                    : in[linearIndex3D(row, col, r1, nRows, nCols)];

    const float s_p =   (AXIS == 0) ? in[linearIndex3D(r2, col, slice, nRows, nCols)]
                      : (AXIS == 1) ? in[linearIndex3D(row, r2, slice, nRows, nCols)]
                                    : in[linearIndex3D(row, col, r2, nRows, nCols)];

    const float s_0 = in[idx];
    out[idx]        = fmaf(-2.0f, s_0, s_m + s_p); // / sigma2;
}

// ============================================================================
//           ROBUST EIGEN-SOLVER FOR 3×3 SYMMETRIC MATRICES (float out)
// ============================================================================
__device__ __host__ __forceinline__
void symmetricEigenvalues3x3(float  a11, float  a22, float  a33,
                             float  a12, float  a13, float  a23,
                             double& l1,  double& l2,  double& l3) noexcept
{
    // Promote to double for accuracy
    const double A11 = static_cast<double>(a11);
    const double A22 = static_cast<double>(a22);
    const double A33 = static_cast<double>(a33);
    const double A12 = static_cast<double>(a12);
    const double A13 = static_cast<double>(a13);
    const double A23 = static_cast<double>(a23);

    // 1) Compute mean
    const double q = (A11 + A22 + A33) / 3.0;

    // 2) Compute centred matrix B = A - q*I
    const double B11 = A11 - q;
    const double B22 = A22 - q;
    const double B33 = A33 - q;

    // 3) Compute invariant p
    const double p2 = (B11*B11 + B22*B22 + B33*B33 +
                       2.0*(A12*A12 + A13*A13 + A23*A23)) / 6.0;

    const double p = sqrt(p2);

    // If p == 0, the matrix is proportional to identity – degenerate case
    if (p < 1e-15) // or std::numeric_limits<double>::epsilon(), but safe for GPU
    {
        l1 = l2 = l3 = q;
        return;
    }

    // 4) Build normalised matrix C = (1/p) * B
    const double C11 = B11 / p;
    const double C22 = B22 / p;
    const double C33 = B33 / p;
    const double C12 = A12 / p;
    const double C13 = A13 / p;
    const double C23 = A23 / p;

    // 5) Compute determinant of C
    const double detC =
        C11*(C22*C33 - C23*C23) -
        C12*(C12*C33 - C23*C13) +
        C13*(C12*C23 - C22*C13);

    // 6) Compute the angle for the eigenvalues
    const double r   = fmax(fmin(detC / 2.0, 1.0), -1.0); // clamp to [-1,1]
    const double phi = acos(r) / 3.0;

    // 7) Eigenvalues of A
    const double eig1 = q + 2.0 * p * cos(phi);
    const double eig3 = q + 2.0 * p * cos(phi + (2.0 * M_PI / 3.0));
    const double eig2 = 3.0*q - eig1 - eig3; // since trace(A) = eig1 + eig2 + eig3

    // 8) Sort by absolute value |λ1| ≤ |λ2| ≤ |λ3| (needed by Frangi)
    double e[3] = {eig1, eig2, eig3};
    int idx[3] = {0, 1, 2};
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (fabs(e[idx[i]]) > fabs(e[idx[j]])) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    l1 = e[idx[0]];
    l2 = e[idx[1]];
    l3 = e[idx[2]];
}

// ============================================================================
//                       VESSELNESS COMPUTATION
// ============================================================================
__global__ void vesselness3D(const float* __restrict__ Dxx,
                             const float* __restrict__ Dyy,
                             const float* __restrict__ Dzz,
                             const float* __restrict__ Dxy,
                             const float* __restrict__ Dxz,
                             const float* __restrict__ Dyz,
                             float*       __restrict__ V,
                             size_t nElem,
                             float  alpha, float beta, float gamma, float sigma2,
                             bool   bright)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nElem) return;

    double l1, l2, l3;
    symmetricEigenvalues3x3(Dxx[idx], Dyy[idx], Dzz[idx],
                            Dxy[idx], Dxz[idx], Dyz[idx],
                            l1, l2, l3);

    const bool keep = bright ? (l2 < 0.0 && l3 < 0.0)
                             : (l2 > 0.0 && l3 > 0.0);

    double v = 0.0;
    if (keep)
    {
        const double a1 = fabs(l1);
        const double a2 = fmax(fabs(l2), static_cast<double>(FLT_EPSILON));
        const double a3 = fmax(fabs(l3), static_cast<double>(FLT_EPSILON));

        const double Ra2 = (a2 * a2) / (a3 * a3);
        const double Rb2 = (a1 * a1) / (a2 * a3);
        const double  S2 = a1*a1 + a2*a2 + a3*a3;

        const double twoAlpha2 = 2.0 * static_cast<double>(alpha) * static_cast<double>(alpha);
        const double twoBeta2  = 2.0 * static_cast<double>(beta)  * static_cast<double>(beta);
        const double twoGamma2 = 2.0 * static_cast<double>(gamma) * static_cast<double>(gamma);

        v = (1.0 - exp(-Ra2 / twoAlpha2)) *
                   exp(-Rb2 / twoBeta2)   *
            (1.0 - exp(-S2  / twoGamma2));

        //v *= sigma2;
    }
    V[idx] = static_cast<float>(v);
}


// ============================================================================
//                            HESSIAN-OF-GAUSSIAN
// ============================================================================
static void hessian3D_gpu(const float* input_d,
                          int nRows, int nCols, int nSlices,
                          float sigma,
                          float* Dxx_d, float* Dyy_d, float* Dzz_d,
                          float* Dxy_d, float* Dxz_d, float* Dyz_d,
                          float* tmp1_d, float* tmp2_d)
{
    // 1) build 1-D kernels
    int kSize = static_cast<int>(ceilf(GAUSSIAN_SIGMA_MULT * sigma));
    if (!(kSize & 1)) ++kSize; // make odd

    std::vector<float> hG, hGx, hGxx;
    makeGaussianKernels1D(hG, hGx, kSize, sigma);
    makeGaussianSecondDerivKernel1D(hGxx, kSize, sigma);

    float *dG, *dGx, *dGxx;
    cudaCheck(cudaMalloc(&dG,   kSize * sizeof(float)));
    cudaCheck(cudaMalloc(&dGx,  kSize * sizeof(float)));
    cudaCheck(cudaMalloc(&dGxx, kSize * sizeof(float)));
    cudaCheck(cudaMemcpy(dG,   hG.data(),   kSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dGx,  hGx.data(),  kSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dGxx, hGxx.data(), kSize * sizeof(float), cudaMemcpyHostToDevice));

    // 2) Dxx = conv1d_x(G''), conv1d_y(G), conv1d_z(G)
    launchConvX(input_d, tmp1_d, dGxx, kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dxx_d,  dG,   kSize, nRows, nCols, nSlices);

    // 3) Dyy = conv1d_x(G), conv1d_y(G''), conv1d_z(G)
    launchConvX(input_d, tmp1_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dGxx, kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dyy_d,  dG,   kSize, nRows, nCols, nSlices);

    // 4) Dzz = conv1d_x(G), conv1d_y(G), conv1d_z(G'')
    launchConvX(input_d, tmp1_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dzz_d,  dGxx, kSize, nRows, nCols, nSlices);

    // 5) Dxy = conv1d_x(G'), conv1d_y(G'), conv1d_z(G)
    launchConvX(input_d, tmp1_d, dGx,  kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dGx,  kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dxy_d,  dG,   kSize, nRows, nCols, nSlices);

    // 6) Dxz = conv1d_x(G'), conv1d_y(G), conv1d_z(G')
    launchConvX(input_d, tmp1_d, dGx,  kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dxz_d,  dGx,  kSize, nRows, nCols, nSlices);

    // 7) Dyz = conv1d_x(G), conv1d_y(G'), conv1d_z(G')
    launchConvX(input_d, tmp1_d, dG,   kSize, nRows, nCols, nSlices);
    launchConvY(tmp1_d, tmp2_d, dGx,  kSize, nRows, nCols, nSlices);
    launchConvZ(tmp2_d, Dyz_d,  dGx,  kSize, nRows, nCols, nSlices);

    cudaCheck(cudaGetLastError());
    cudaFree(dG);
    cudaFree(dGx);
    cudaFree(dGxx);
}

// ============================================================================
//                        MAX PROJECTION WITH THRESHOLD
// ============================================================================
__global__ void maxInPlace(const float* src, float* dst, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fmaxf(dst[i], src[i]);   // <-- just a max
}

__global__ void scaleHessianUpKernel(
    //float* Dxx, float* Dyy, float* Dzz,
    float* Dxy, float* Dxz, float* Dyz, size_t nElem, float sigma2)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElem) return;
    //Dxx[i] *= sigma2;
    //Dyy[i] *= sigma2;
    //Dzz[i] *= sigma2;
    Dxy[i] *= sigma2;
    Dxz[i] *= sigma2;
    Dyz[i] *= sigma2;
}

// ============================================================================
//                          MEX ENTRY POINT
// ============================================================================
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 8)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Usage: out = fibermetric_gpu(gpuArray_single, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, objectPolarity, structureSensitivity)");

    mxInitGPU();

    // Parse inputs
    const mxGPUArray* inGpu = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(inGpu) != mxSINGLE_CLASS ||
        mxGPUGetNumberOfDimensions(inGpu) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type",
                          "Input must be a 3-D single-precision gpuArray.");

    const mwSize* dims   = mxGPUGetDimensions(inGpu);
    const int nRows  = static_cast<int>(dims[0]);
    const int nCols  = static_cast<int>(dims[1]);
    const int nSlc   = static_cast<int>(dims[2]);
    const size_t nElem  = static_cast<size_t>(nRows) * nCols * nSlc;
    const float* img_d  = static_cast<const float*>(mxGPUGetDataReadOnly(inGpu));

    const float sigmaFrom = static_cast<float>(mxGetScalar(prhs[1]));
    const float sigmaTo   = static_cast<float>(mxGetScalar(prhs[2]));
    const float sigmaStep = static_cast<float>(mxGetScalar(prhs[3]));
    const float alpha     = static_cast<float>(mxGetScalar(prhs[4]));
    const float beta      = static_cast<float>(mxGetScalar(prhs[5]));

    char polStr[16];
    mxGetString(prhs[6], polStr, sizeof(polStr));
    const bool brightPolarity = (std::strcmp(polStr, "bright") == 0);

    // structureSensitivity is required, but you can default if desired
    float structureSensitivity = (nrhs >= 8) ? static_cast<float>(mxGetScalar(prhs[7])) : 0.5f;

    // --- Compute thresh internally
    const float thresh = 0.2f * structureSensitivity;

    // Output allocation
    mxGPUArray* outGpu = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* out_d = static_cast<float*>(mxGPUGetData(outGpu));
    cudaCheck(cudaMemset(out_d, 0, nElem * sizeof(float)));

    // Work buffers
    float *tmp1_d, *tmp2_d, *Dxx_d, *Dyy_d, *Dzz_d,
          *Dxy_d, *Dxz_d, *Dyz_d, *V_d;
    const size_t BUF = nElem * sizeof(float);
    cudaCheck(cudaMalloc(&tmp1_d, BUF));
    cudaCheck(cudaMalloc(&tmp2_d, BUF));
    cudaCheck(cudaMalloc(&Dxx_d,  BUF));
    cudaCheck(cudaMalloc(&Dyy_d,  BUF));
    cudaCheck(cudaMalloc(&Dzz_d,  BUF));
    cudaCheck(cudaMalloc(&Dxy_d,  BUF));
    cudaCheck(cudaMalloc(&Dxz_d,  BUF));
    cudaCheck(cudaMalloc(&Dyz_d,  BUF));
    cudaCheck(cudaMalloc(&V_d,    BUF));

    const int tpb = 256;
    const int gpb = static_cast<int>((nElem + tpb - 1) / tpb);

    // Build sigma vector
    std::vector<float> sigmas;
    for (float sigma = sigmaFrom; sigma <= sigmaTo + 1e-4f; sigma += sigmaStep)
        sigmas.push_back(sigma);

    for (float& sigma : sigmas)
    {
        hessian3D_gpu(img_d,
                      nRows, nCols, nSlc,
                      sigma,
                      Dxx_d, Dyy_d, Dzz_d,
                      Dxy_d, Dxz_d, Dyz_d,
                      tmp1_d, tmp2_d);

        float sigma2 = sigma * sigma;
        scaleHessianUpKernel<<<gpb, tpb>>>(Dxy_d, Dxz_d, Dyz_d, nElem, sigma2);
        cudaCheck(cudaGetLastError());

        const float gamma = structureSensitivity * sigma;
        cudaCheck(cudaMemset(V_d, 0, BUF));
        vesselness3D<<<gpb, tpb>>>(Dxx_d, Dyy_d, Dzz_d,
                                   Dxy_d, Dxz_d, Dyz_d,
                                   V_d, nElem,
                                   alpha, beta, gamma, sigma2,
                                   brightPolarity);
        cudaCheck(cudaGetLastError());

        if (sigmas.size() > 1) {
            maxInPlace<<<gpb,tpb>>>(V_d, out_d, nElem);
            cudaCheck(cudaGetLastError());
        } else {
            cudaCheck(cudaMemcpy(out_d, V_d, nElem * sizeof(float), cudaMemcpyDeviceToDevice));
            cudaCheck(cudaGetLastError());
        }
    }
    plhs[0] = mxGPUCreateMxArrayOnGPU(outGpu);

    // Cleanup
    cudaFree(tmp1_d);  cudaFree(tmp2_d);
    cudaFree(Dxx_d);   cudaFree(Dyy_d);   cudaFree(Dzz_d);
    cudaFree(Dxy_d);   cudaFree(Dxz_d);   cudaFree(Dyz_d);
    cudaFree(V_d);
    mxGPUDestroyGPUArray(inGpu);
    mxGPUDestroyGPUArray(outGpu);
}