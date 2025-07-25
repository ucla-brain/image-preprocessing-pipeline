#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <vector>
#include <cstring>

#if defined(_WIN32)
    #define STRICMP _stricmp
#else
    #include <strings.h>  // For strcasecmp
    #define STRICMP strcasecmp
#endif

constexpr double SQRT_TWO_PI = 2.5066282746310002; // sqrt(2π)
constexpr float TWO_PI_OVER_THREE = 2.0943951023931953f;
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

__device__ __host__ __forceinline__
int reflectCoord(int p, int len) noexcept
{
    // Handle degenerate axes (len == 0 never occurs, len == 1 is common for 2-D data)
    if (len <= 1) return 0;

    // Mirror-padding with period 2*len-2  (…len-2, len-1, len-2, …, 1, 0, 1, …)
    int period = 2 * len - 2;
    p = ((p % period) + period) % period;   // wrap into [0, period-1]
    if (p >= len)                           // reflect the second half
        p = period - p;
    return p;                               // guaranteed 0 ≤ p < len
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
    const float* __restrict__ input,  float* __restrict__ output,
    const float* __restrict__ kernelDev, int kernelLen,
    int nRows, int nCols, int nSlices)
{
    size_t idx    = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total  = size_t(nRows) * nCols * nSlices;
    if (idx >= total) return;

    int row   = int(idx % nRows);
    int col   = int((idx / nRows) % nCols);
    int slice = int(idx / (size_t(nRows) * nCols));

    int halfWidth = kernelLen >> 1;
    float acc = 0.f;

    for (int k = -halfWidth; k <= halfWidth; ++k)
    {
        int rr = row, cc = col, ss = slice;   // start at current voxel
        if (AXIS == 0) rr += k;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;

        // NEW: robust mirror padding for any dimension length (including 1)
        rr = reflectCoord(rr, nRows);
        cc = reflectCoord(cc, nCols);
        ss = reflectCoord(ss, nSlices);

        acc = fmaf(kernelDev[k + halfWidth],
                   input[linearIndex3D(rr, cc, ss, nRows, nCols)],
                   acc);
    }
    output[idx] = acc;
}

void launchSeparableConvolutionDevice(
    int axis, const float* inputDev, float* outputDev, const float* kernelDev,
    int kernelLen, int nRows, int nCols, int nSlices, int threadsPerBlock, cudaStream_t stream)
{
    size_t total = size_t(nRows) * nCols * nSlices;
    size_t nBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    if (axis == 0)
        separableConvolution1DDeviceKernelFlat<0><<<nBlocks, threadsPerBlock, 0, stream>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else if (axis == 1)
        separableConvolution1DDeviceKernelFlat<1><<<nBlocks, threadsPerBlock, 0, stream>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
    else
        separableConvolution1DDeviceKernelFlat<2><<<nBlocks, threadsPerBlock, 0, stream>>>(
            inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);

    cudaCheck(cudaPeekAtLastError());
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
    int nRows, int nCols, int nSlices, cudaStream_t stream)
{
    dim3 bx(32,1,1), gx((nRows+31)/32, nCols, nSlices);
    secondDerivative5ptKernel<0><<<gx, bx, 0, stream>>>(smoothedInput, Dxx, nRows, nCols, nSlices);
    dim3 by(1,32,1), gy(nRows, (nCols+31)/32, nSlices);
    secondDerivative5ptKernel<1><<<gy, by, 0, stream>>>(smoothedInput, Dyy, nRows, nCols, nSlices);
    dim3 bz(1,1,32), gz(nRows, nCols, (nSlices+31)/32);
    secondDerivative5ptKernel<2><<<gz, bz, 0, stream>>>(smoothedInput, Dzz, nRows, nCols, nSlices);
    cudaCheck(cudaGetLastError());
}

//---------------- Cross Derivatives: chain of separable convs (reuse buffers) ----------------
void launchCrossDerivativesDevice(
    const float* inputDev, float* Dxy, float* Dxz, float* Dyz,
    float* tmp1, float* tmp2,
    const float* derivKernelDev, const float* gaussKernelDev, int kernelLen,
    int nRows, int nCols, int nSlices, int threadsPerBlock, cudaStream_t stream)
{
    // Dxy: d2/dxdy
    launchSeparableConvolutionDevice(0, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(1, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(2, tmp2, Dxy, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    // Dxz: d2/dxdz
    launchSeparableConvolutionDevice(0, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(2, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(1, tmp2, Dxz, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    // Dyz: d2/dydz
    launchSeparableConvolutionDevice(1, inputDev, tmp1, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(2, tmp1, tmp2, derivKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
    launchSeparableConvolutionDevice(0, tmp2, Dyz, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
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

//-------------------- Vesselness Kernels -------------
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

__global__ void neuritenessMeijeringKernelFromEigen(
    const float* __restrict__ l1,
    const float* __restrict__ l2,
    const float* __restrict__ l3,
    float* __restrict__ neuriteness,
    size_t n,
    bool bright)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ev1 = l1[idx], ev2 = l2[idx], ev3 = l3[idx];

    // Eigenvalues are assumed sorted by absolute value: |ev1| <= |ev2| <= |ev3|
    bool isNeurite = bright ? (ev2 < 0.f && ev3 < 0.f)
                            : (ev2 > 0.f && ev3 > 0.f);

    neuriteness[idx] = isNeurite ? fabsf(ev1) : 0.f;
}

__global__ void vesselnessJermanKernelFromEigen(
    const float* __restrict__ l1, const float* __restrict__ l2, const float* __restrict__ l3,
    float* __restrict__ vesselness, size_t n,
    const float inv2Alpha2, const float inv2Beta2,
    bool bright, const float scaleNorm)
{
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val1 = l1[idx], val2 = l2[idx], val3 = l3[idx];
    bool condition = bright ? (val2 < 0.f && val3 < 0.f) : (val2 > 0.f && val3 > 0.f);
    float response = 0.f;
    if (condition) {
        float absL1 = fabsf(val1), absL2 = fabsf(val2), absL3 = fabsf(val3);
        float Ra2 = (absL2 * absL2) / (absL3 * absL3 + EPSILON);                  // Plate/tube
        float Rb2 = (absL1 * absL1) / fmaf(absL2, absL3, EPSILON); // Blob/tube
        // Jerman: (1 - exp(-Ra^2/2α^2)) * exp(-Rb^2/2β^2)
        float expRa = __expf(-Ra2 * inv2Alpha2);
        float expRb = __expf(-Rb2 * inv2Beta2);
        response = fmaf(-expRa, expRb, expRb);
        response *= scaleNorm;
    }
    vesselness[idx] = response;
}

//--------------------utility kernels

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
            "Usage: fibermetric_gpu(gpuArraySingle3D, sigmaFrom, sigmaTo, sigmaStep, alpha, beta, structureSensitivity, 'bright'|'dark', 'frangi'|'sato'|'meijering'|'jerman')");

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
    size_t n = (size_t)nRows * (size_t)nCols * (size_t)nSlices;
    const float* inputDev = static_cast<const float*>(mxGPUGetDataReadOnly(input));

    double sigmaFrom = mxGetScalar(prhs[1]);
    double sigmaTo   = mxGetScalar(prhs[2]);
    double sigmaStep = mxGetScalar(prhs[3]);
    float alpha      = float(mxGetScalar(prhs[4]));
    float beta       = float(mxGetScalar(prhs[5]));
    float structureSensitivity = float(mxGetScalar(prhs[6]));
    char polarityBuf[16]; mxGetString(prhs[7], polarityBuf, sizeof(polarityBuf));
    bool bright = (STRICMP(polarityBuf, "bright") == 0);
    char methodBuf[16]; mxGetString(prhs[8], methodBuf, sizeof(methodBuf));
    bool useFrangi = (STRICMP(methodBuf, "frangi") == 0);
    bool useSato   = (STRICMP(methodBuf, "sato") == 0);
    bool useMeijering = (STRICMP(methodBuf, "meijering") == 0);
    bool useJerman = (STRICMP(methodBuf, "jerman") == 0);
    if (!useFrangi && !useSato && !useMeijering && !useJerman)
        mexErrMsgIdAndTxt("fibermetric_gpu:usage",
            "Last argument must be 'frangi', 'sato', 'meijering', or 'jerman'.");

    //--- Output Allocation ---
    mxGPUArray* output = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(output));
    size_t bytes = n * sizeof(float);

    //--- Workspace Buffers ---
    float *tmp1, *tmp2;
    cudaCheck(cudaMalloc(&tmp1, bytes));
    cudaCheck(cudaMalloc(&tmp2, bytes));

    // --- Allocate Hessian and eigenvalue arrays ---
    float *Dxx, *Dyy, *Dzz, *Dxy, *Dxz, *Dyz, *l1, *l2, *l3;
    cudaCheck(cudaMalloc(&Dxx, bytes));
    cudaCheck(cudaMalloc(&Dyy, bytes));
    cudaCheck(cudaMalloc(&Dzz, bytes));
    cudaCheck(cudaMalloc(&Dxy, bytes));
    cudaCheck(cudaMalloc(&Dxz, bytes));
    cudaCheck(cudaMalloc(&Dyz, bytes));
    cudaCheck(cudaMalloc(&l1, bytes));
    cudaCheck(cudaMalloc(&l2, bytes));
    cudaCheck(cudaMalloc(&l3, bytes));

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

    // --- CUDA Graph Variables ---
    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreate(&stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    //---- **Key fix: Zero output buffer before projection** ----
    cudaCheck(cudaMemsetAsync(outputDev, 0, n * sizeof(float), stream));
    cudaCheck(cudaStreamSynchronize(stream));

    //--- Main Loop Over Scales ---
    for (size_t sigmaIdx = 0; sigmaIdx < sigmaList.size(); ++sigmaIdx) {
        double sigma = sigmaList[sigmaIdx];

        // -- Create Gaussian and derivative kernels for this sigma --
        float* gaussKernelDev = nullptr;
        float* derivKernelDev = nullptr;
        int halfWidth = int(std::ceil(GAUSS_HALFWIDTH_MULT * sigma));
        int kernelLen = 2 * halfWidth + 1;
        buildGaussianAndDerivativeKernelsDevice(gaussKernelDev, derivKernelDev, kernelLen, sigma, threadsPerBlock);

        // --- (Re)calculate constants as needed ---
        float sigmaSq = float(sigma * sigma);
        const float inv2Alpha2 = static_cast<float>(1.0 / (2.0 * double(alpha) * double(alpha)));
        const float inv2Beta2  = static_cast<float>(1.0 / (2.0 * double(beta)  * double(beta )));
        float gamma = structureSensitivity * float(sigma);
        const float inv2Gamma2 = static_cast<float>(1.0 / (2.0 * double(gamma) * double(gamma)));

        // --- Clean up any old graph (always recapture per sigma for correctness) ---
        if (graphExec) {
            cudaCheck(cudaGraphExecDestroy(graphExec));
            graphExec = nullptr;
        }
        if (graph) {
            cudaCheck(cudaGraphDestroy(graph));
            graph = nullptr;
        }

        // ===================== Graph Capture ==========================
        cudaCheck(cudaStreamSynchronize(stream));
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // 1. Gaussian smoothing (separable, X, Y, Z)
        launchSeparableConvolutionDevice(0, inputDev, tmp1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
        launchSeparableConvolutionDevice(1, tmp1, tmp2, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);
        launchSeparableConvolutionDevice(2, tmp2, tmp1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);

        // 2. Hessian diagonals
        launchSecondDerivatives(tmp1, Dxx, Dyy, Dzz, nRows, nCols, nSlices, stream);

        // 3. Cross-derivatives
        launchCrossDerivativesDevice(inputDev, Dxy, Dxz, Dyz, tmp1, tmp2,
            derivKernelDev, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream);

        // 4. Scale normalization
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dxx, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dyy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dzz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dxy, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dxz, n, sigmaSq);
        scaleArrayInPlaceKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(Dyz, n, sigmaSq);

        // 5. Eigenvalue decomposition
        hessianToEigenvaluesKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(
            Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, l1, l2, l3, n);

        // 6. Vesselness kernel (choose the right one for your method)
        if (useFrangi) {
            vesselnessFrangiKernelFromEigen<<<nBlocks, threadsPerBlock, 0, stream>>>(
                l1, l2, l3, tmp1, n, inv2Alpha2, inv2Beta2, inv2Gamma2, bright, doScaleNorm, scaleNorm);
        } else if (useSato) {
            vesselnessSatoKernelFromEigen<<<nBlocks, threadsPerBlock, 0, stream>>>(
                l1, l2, l3, tmp1, n, inv2Alpha2, inv2Beta2, bright);
        } else if (useMeijering) {
            neuritenessMeijeringKernelFromEigen<<<nBlocks, threadsPerBlock, 0, stream>>>(
                l1, l2, l3, tmp1, n, bright);
        } else if (useJerman) {
            vesselnessJermanKernelFromEigen<<<nBlocks, threadsPerBlock, 0, stream>>>(
                l1, l2, l3, tmp1, n, inv2Alpha2, inv2Beta2, bright, scaleNorm);
        }

        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // --- Launch the captured graph for this sigma ---
        cudaCheck(cudaGraphLaunch(graphExec, stream));
        cudaCheck(cudaStreamSynchronize(stream)); // Wait for results

        // --- Max projection or copy, as before ---
        if (!singleSigma)
            elementwiseMaxKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(tmp1, outputDev, n);
        else
            cudaCheck(cudaMemcpyAsync(outputDev, tmp1, bytes, cudaMemcpyDeviceToDevice, stream));

        cudaCheck(cudaStreamSynchronize(stream)); // Ensure all results are ready before output

        cudaFree(gaussKernelDev);
        cudaFree(derivKernelDev);
    }

    // Cleanup: (after the sigma loop)
    if (graphExec) cudaCheck(cudaGraphExecDestroy(graphExec));
    if (graph)     cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));

    //--- Output ---
    cudaCheck(cudaDeviceSynchronize());
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);

    //--- Free device buffers ---
    cudaFree(tmp1); cudaFree(tmp2);
    cudaFree(Dxx); cudaFree(Dyy); cudaFree(Dzz);
    cudaFree(Dxy); cudaFree(Dxz); cudaFree(Dyz);
    cudaFree(l1); cudaFree(l2); cudaFree(l3);

    mxGPUDestroyGPUArray(input); mxGPUDestroyGPUArray(output);
}
