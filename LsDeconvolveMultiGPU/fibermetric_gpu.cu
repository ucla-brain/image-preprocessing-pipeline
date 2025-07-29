/*
 * ===============================  Jerman Mode Parameter Mapping  ===============================
 *
 * When vesselness method is set to 'jerman', the parameters alpha and beta are **reinterpreted**:
 *
 *   - alpha: used as Jerman's tau (tau, 0.5–1), controls lambda3 truncation
 *   - beta : used as Jerman's C (C > 0), guesses lambda3min as -C*sigma^2
 *
 * For Frangi/Sato/Meijering, alpha and beta retain their classic meanings.
 *
 * structureSensitivity: Always interpreted as eigenvalue magnitude thresholding
 *
 * The output is **not thresholded** on the GPU for Jerman. Thresholding and any further
 * post-processing must be done in MATLAB.
 *
 * ================================================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

#include <cfloat>
#include <cmath>
#include <vector>
#include <cstring>

#if defined(_WIN32)
    #define STRICMP _stricmp
#else
    #include <strings.h>  // For strcasecmp
    #define STRICMP strcasecmp
#endif

#define MAX_CONST_COEF_LEN 4096
__constant__ float gCoef[MAX_CONST_COEF_LEN];

constexpr double SQRT_TWO_PI             = 2.5066282746310002;  // sqrt(2π)
constexpr float TWO_PI_OVER_THREE        = 2.0943951023931953f; // 2π/3
constexpr float ONE_OVER_SIX             = 0.1666666666666667f; // 1/6
constexpr float GAUSS_HALFWIDTH_MULT     = 8.0f;
constexpr float EPSILON                  = 1e-7f;
constexpr int THREADS_PER_BLOCK          = 1024;
constexpr double FINITE_DIFF_5PT_DIVISOR = 12.0;

// -- Hardware shared memory limits --
// Most NVIDIA GPUs provide at least 48KB (49152 bytes) shared memory per block.
// Some allow up to 96KB (98304 bytes), but 48KB is the safe minimum for compatibility.
// DEFAULT_TILE_SIZE + kernelLen - 1 must satisfy: (DEFAULT_TILE_SIZE + kernelLen - 1) * sizeof(float) <= SHARED_MEMORY_PER_BLOCK
constexpr int SHARED_MEMORY_PER_BLOCK = 48 * 1024; // 48 KB safe default for all compute >= 3.0
constexpr int DEFAULT_TILE_SIZE = 256;  // Can tune for best occupancy. Set to 128 or 256 as per shared memory limit and occupancy

static_assert((DEFAULT_TILE_SIZE + MAX_CONST_COEF_LEN - 1) * sizeof(float) <= SHARED_MEMORY_PER_BLOCK,
        "DEFAULT_TILE_SIZE + kernelLen - 1 exceeds hardware shared memory per block. Lower DEFAULT_TILE_SIZE or kernelLen, or target a GPU with more shared memory.");

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
__global__ void generateGaussianKernel(
    float* __restrict__ gaussKernel,
    int kernelLen, int half, double sigma, double sigmaSq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kernelLen) return;
    int x = i - half;
    double norm = 1.0 / (SQRT_TWO_PI * sigma);
    double g = norm * exp(-0.5 * (x * x) / sigmaSq);
    gaussKernel[i] = static_cast<float>(g);
}

void buildGaussianKernelDevice(
    float*& gaussKernelDev,
    int kernelLen, double sigma, double sigmaSq, int threadsPerBlock)
{
    size_t kernelBytes = size_t(kernelLen) * sizeof(float);
    cudaCheck(cudaMalloc(&gaussKernelDev, kernelBytes));
    int blocks = (kernelLen + threadsPerBlock - 1) / threadsPerBlock;
    generateGaussianKernel<<<blocks, threadsPerBlock>>>(gaussKernelDev, kernelLen, kernelLen / 2, sigma, sigmaSq);
    cudaCheck(cudaGetLastError());
}

//---------------- Separable 1D Convolution (fmaf, in-place option) -------------------------
template<int AXIS>
__global__ void separableConvolution1DTiledConstKernel(
    const float* __restrict__ input,  float* __restrict__ output,
    int kernelLen, int nRows, int nCols, int nSlices, int tileSize)
{
    // Only AXIS == 0 (X-pass) is supported for shared-mem tiling here
    extern __shared__ float tile[];
    int col = blockIdx.y;
    int slice = blockIdx.z;

    int halfWidth = kernelLen >> 1;
    int tileStart = blockIdx.x * tileSize;
    int globalY = col, globalZ = slice;

    // Each thread loads one element, plus halo at start/end
    for (int tx = threadIdx.x; tx < tileSize + kernelLen - 1; tx += blockDim.x)
    {
        int inputX = tileStart + tx - halfWidth;
        // Reflect boundary
        int rr = reflectCoord(inputX, nRows);

        tile[tx] = input[linearIndex3D(rr, globalY, globalZ, nRows, nCols)];
    }
    __syncthreads();

    int x = tileStart + threadIdx.x;
    if (x >= nRows) return;

    float acc = 0.f;
    for (int k = -halfWidth; k <= halfWidth; ++k)
    {
        int idx = threadIdx.x + k + halfWidth;
        acc = fmaf(gCoef[k + halfWidth], tile[idx], acc);
    }
    output[linearIndex3D(x, globalY, globalZ, nRows, nCols)] = acc;
}

// -- Constant-memory, untiled, for Y/Z (AXIS=1,2) --
template<int AXIS>
__global__ void separableConvolution1DConstKernel(
    const float* __restrict__ input,  float* __restrict__ output,
    int kernelLen, int nRows, int nCols, int nSlices)
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
        int rr = row, cc = col, ss = slice;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;
        cc = reflectCoord(cc, nCols);
        ss = reflectCoord(ss, nSlices);
        acc = fmaf(gCoef[k + halfWidth], input[linearIndex3D(rr, cc, ss, nRows, nCols)], acc);
    }
    output[idx] = acc;
}

template<int AXIS>
__global__ void separableConvolution1DDeviceKernelFlat(
    const float* __restrict__ input,  float* __restrict__ output,
    const float* __restrict__ kernelDev, int kernelLen,
    int nRows, int nCols, int nSlices)
{
    size_t idx    = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total  = size_t(nRows) * nCols * nSlices;
    if (idx >= total) return;

#if (AXIS == 0)
    int col   = int((idx / nRows) % nCols);
    int slice = int(idx / (size_t(nRows) * nCols));
    // Shared memory tiling for X axis (rows)
    extern __shared__ float shInput[];
    int halfWidth = kernelLen >> 1;
    int tileStart = blockIdx.x * blockDim.x - halfWidth;
    int localIdx  = threadIdx.x + halfWidth;
    // Load the whole tile (including halo) into shared memory
    for (int k = threadIdx.x; k < blockDim.x + 2*halfWidth; k += blockDim.x) {
        int globalRow = tileStart + k;
        int safeRow = reflectCoord(globalRow, nRows);
        shInput[k] = input[linearIndex3D(safeRow, col, slice, nRows, nCols)];
    }
    __syncthreads();

    // Now do convolution in shared memory
    float acc = 0.f;
    for (int k = -halfWidth; k <= halfWidth; ++k)
    {
        int sIdx = localIdx + k;
        float coef = kernelDev[k + halfWidth];
        acc = fmaf(coef, shInput[sIdx], acc);
    }
    output[idx] = acc;
#else
    int row   = int(idx % nRows);
    int col   = int((idx / nRows) % nCols);
    int slice = int(idx / (size_t(nRows) * nCols));
    // Y or Z axis: original device path
    int halfWidth = kernelLen >> 1;
    float acc = 0.f;
    for (int k = -halfWidth; k <= halfWidth; ++k)
    {
        int rr = row, cc = col, ss = slice;
        if (AXIS == 1) cc += k;
        if (AXIS == 2) ss += k;
        cc = reflectCoord(cc, nCols);
        ss = reflectCoord(ss, nSlices);
        acc = fmaf(kernelDev[k + halfWidth], input[linearIndex3D(rr, cc, ss, nRows, nCols)], acc);
    }
    output[idx] = acc;
#endif
}

void launchSeparableConvolutionDevice(
    int axis, const float* inputDev, float* outputDev, const float* kernelDev,
    int kernelLen, int nRows, int nCols, int nSlices, int threadsPerBlock,
    cudaStream_t stream, bool useConstMem, int tileSize)
{
    size_t total = size_t(nRows) * nCols * nSlices;
    size_t nBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    if (useConstMem && kernelLen <= MAX_CONST_COEF_LEN)
    {
        if (axis == 0)
        {
            int shmemBytes = (tileSize + kernelLen - 1) * sizeof(float);

            int maxSharedMem = SHARED_MEMORY_PER_BLOCK;
            // Query actual device limit at runtime
            cudaDeviceProp prop;
            cudaCheck(cudaGetDeviceProperties(&prop, 0));
            maxSharedMem = prop.sharedMemPerBlock;

            if (shmemBytes > maxSharedMem) {
                mexErrMsgIdAndTxt("fibermetric_gpu:sharedmem",
                    "Requested shared memory per block (%d bytes) exceeds device limit (%d bytes). "
                    "Reduce DEFAULT_TILE_SIZE or kernel size.", shmemBytes, maxSharedMem);
            }

            dim3 blockDim(tileSize, 1, 1);
            dim3 gridDim((nRows + tileSize - 1) / tileSize, nCols, nSlices);
            size_t shmemSize = (tileSize + kernelLen - 1) * sizeof(float);
            separableConvolution1DTiledConstKernel<0><<<gridDim, blockDim, shmemSize, stream>>>(
                inputDev, outputDev, kernelLen, nRows, nCols, nSlices, tileSize);
        }
        else if (axis == 1)
        {
            separableConvolution1DConstKernel<1><<<nBlocks, threadsPerBlock, 0, stream>>>(
                inputDev, outputDev, kernelLen, nRows, nCols, nSlices);
        }
        else // axis == 2
        {
            separableConvolution1DConstKernel<2><<<nBlocks, threadsPerBlock, 0, stream>>>(
                inputDev, outputDev, kernelLen, nRows, nCols, nSlices);
        }
    }
    else
    {
        if (axis == 0)
        {
            separableConvolution1DDeviceKernelFlat<0><<<nBlocks, threadsPerBlock, 0, stream>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
        }
        else if (axis == 1)
        {
            separableConvolution1DDeviceKernelFlat<1><<<nBlocks, threadsPerBlock, 0, stream>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
        }
        else // axis == 2
        {
            separableConvolution1DDeviceKernelFlat<2><<<nBlocks, threadsPerBlock, 0, stream>>>(
                inputDev, outputDev, kernelDev, kernelLen, nRows, nCols, nSlices);
        }
    }
    cudaCheck(cudaPeekAtLastError());
}

void uploadCoefficientsToConstantMemory(const float* h_gCoef, int kernelLen)
{
    cudaCheck(cudaMemcpyToSymbol(gCoef, h_gCoef, kernelLen * sizeof(float), 0, cudaMemcpyHostToDevice));
}

//-------------------- 5-point Second Derivative (fmaf, precompute squares) ----------------------
// Fused Hessian kernel with shared memory padding to avoid bank conflicts

//-------------------- Eigenvalue Decomposition (fmaf, float only) ----------------------
template <typename T>
__device__ __host__ __forceinline__ void swapCUDA(T& a, T& b) noexcept { T tmp = a; a = b; b = tmp; }

enum VesselnessMethod : int { METHOD_FRANGI = 0, METHOD_SATO, METHOD_MEIJERING, METHOD_JERMAN };

__device__ __forceinline__
void computeSymmetricEigenvalues3x3(
    float A11, float A22, float A33, float A12, float A13, float A23,
    float& l1, float& l2, float& l3)
{
    // (same as your current function)
    float q = (A11 + A22 + A33) / 3.f;
    float B11 = A11 - q, B22 = A22 - q, B33 = A33 - q;
    float A12Sq = A12*A12, A13Sq = A13*A13, A23Sq = A23*A23;
    float p2 = fmaf(fmaf(B11, B11, fmaf(B22, B22, fmaf(B33, B33, 2.f * (A12Sq + A13Sq + A23Sq)))), ONE_OVER_SIX, EPSILON);
    float p = sqrtf(p2);
    if (p < 1e-8f) { l1 = l2 = l3 = q; return; }
    float C11 = B11 / p, C22 = B22 / p, C33 = B33 / p;
    float C12 = A12 / p, C13 = A13 / p, C23 = A23 / p;
    float detC = fmaf(C11, fmaf(C22, C33, -C23*C23), fmaf(-C12, fmaf(C12, C33, -C13*C23), C13 * fmaf(C12, C23, -C13*C22)));
    float r = fmaxf(fminf(detC * 0.5f, 1.f), -1.f);
    float phi         = acosf(r) / 3.f;
    float cosPhi      = cosf(phi);
    float cosPhiShift = cosf(TWO_PI_OVER_THREE + phi);
    float twiceP = 2.f * p;
    float x1 = fmaf(twiceP, cosPhi     , q);
    float x3 = fmaf(twiceP, cosPhiShift, q);
    float x2 = fmaf(3.f, q, -x1 - x3);

    float vals[3] = {x1, x2, x3};
    float absVals[3] = {fabsf(x1), fabsf(x2), fabsf(x3)};
    int order[3] = {0, 1, 2};
    if (absVals[0] > absVals[1]) { float t=order[0];order[0]=order[1];order[1]=t;t=absVals[0];absVals[0]=absVals[1];absVals[1]=t; }
    if (absVals[1] > absVals[2]) { float t=order[1];order[1]=order[2];order[2]=t;t=absVals[1];absVals[1]=absVals[2];absVals[2]=t; }
    if (absVals[0] > absVals[1]) { float t=order[0];order[0]=order[1];order[1]=t;t=absVals[0];absVals[0]=absVals[1];absVals[1]=t; }
    l1 = vals[order[0]]; l2 = vals[order[1]]; l3 = vals[order[2]];
}

// --- Vesselness Inline Functions ---

__device__ __forceinline__
float vesselnessFrangi(float l1, float l2, float l3, bool bright, float inv2Alpha2, float inv2Beta2, float inv2Gamma2)
{
    bool condition = bright ? (l2 < 0.f && l3 < 0.f) : (l2 > 0.f && l3 > 0.f);
    if (!condition) return 0.f;
    //float absL1 = fabsf(l1), absL2 = fabsf(l2), absL3 = fabsf(l3);
    float absL2 = fabsf(l2), absL3 = fabsf(l3);
    float Ra2   = (absL2 * absL2) / (absL3 * absL3 + EPSILON);
    //float Rb2   = (absL1 * absL1) / fmaf(absL2, absL3, EPSILON);
    float Rb2   = (l1 * l1) / fmaf(absL2, absL3, EPSILON);
    float S2    = fmaf(l1, l1, fmaf(l2, l2, l3 * l3));
    float expRa = __expf(-Ra2 * inv2Alpha2);
    float expRb = __expf(-Rb2 * inv2Beta2);
    float expS2 = __expf(-S2  * inv2Gamma2);
    float tmp   = fmaf(-expRa, expRb, expRb);
    return fmaf(-expS2, tmp, tmp);
}

__device__ __forceinline__
float vesselnessSato(float l1, float l2, float l3, bool bright, float inv2Alpha2, float inv2Beta2)
{
    bool isVessel = bright ? (l2 < 0.f && l3 < 0.f) : (l2 > 0.f && l3 > 0.f);
    if (!isVessel) return 0.f;
    //float absL1 = fabsf(l1), absL2 = fabsf(l2), absL3 = fabsf(l3);
    float absL2 = fabsf(l2);
    //float expL1 = __expf(-absL1 * absL1 * inv2Alpha2);
    //float expL3 = __expf(-absL3 * absL3 * inv2Beta2);
    float expL1 = __expf(-l1 * l1 * inv2Alpha2);
    float expL3 = __expf(-l3 * l3 * inv2Beta2);
    float absL2TimesExpL1 = absL2 * expL1;
    return fmaf(-expL3, absL2TimesExpL1, absL2TimesExpL1);
}

__device__ __forceinline__
float neuritenessMeijering(float l1, float l2, float l3, bool bright)
{
    bool isNeurite = bright ? (l2 < 0.f && l3 < 0.f) : (l2 > 0.f && l3 > 0.f);
    return isNeurite ? fabsf(l1) : 0.f;
}

// For Jerman: inv2Alpha2= tau, inv2Beta2= lambda3minGuess, inv2Gamma2= structureSensitivity threshold
__device__ __forceinline__
float vesselnessJerman(
    float l1, float l2, float l3,        // Hessian eigenvalues (ordered or not, just be consistent)
    bool bright,                         // true: bright-on-dark, false: dark-on-bright
    float jermanTau,                     // tau parameter (0.5–1), controls truncation
    float lambda3minGuess,               // C * sigma^2, your global min guess for Lambda3
    float structureSensitivity           // replaces hardcoded eigenvalue threshold
)
{
    // === Step 1: Eigenvalue sign test for blob polarity ===
    bool isBlob = bright ? (l1 < 0.f && l2 < 0.f && l3 < 0.f)
                         : (l1 > 0.f && l2 > 0.f && l3 > 0.f);
    if (!isBlob)
        return 0.f;

    // === Step 2: Eigenvalue noise/weak response thresholding ===
    if (fabsf(l1) < structureSensitivity) l1 = 0.f;
    if (fabsf(l2) < structureSensitivity) l2 = 0.f;
    if (fabsf(l3) < structureSensitivity) l3 = 0.f;

    // === Step 3: Truncate lambda3 using tau and lambda3minGuess ===
    float lambda3m;
    if (bright)
        lambda3m = (l3 >= lambda3minGuess * jermanTau) ? (lambda3minGuess * jermanTau) : l3;
    else
        lambda3m = (l3 <= lambda3minGuess * jermanTau) ? (lambda3minGuess * jermanTau) : l3;

    // === Step 4: Main Jerman formula ===
    float denom = powf(fmaf(2.0f, l1, lambda3m), 3.0f); // (2*l1 + lambda3m)^3

    if (!isfinite(denom) || denom == 0.f)
        return 0.f;

    float numer = (l1 * l1) * lambda3m * 27.0f;
    float response = numer / denom;

    // === Step 5: Clamp response for outlier case
    if (fabsf(l1) > fabsf(lambda3m))
        response = 1.0f;

    // === Step 6: Clamp non-finite response to zero
    if (!isfinite(response))
        response = 0.f;

    return response;
}

// ======= Fused Hessian + Vesselness Kernel =======

template<int TILE_X=16, int TILE_Y=8, int TILE_Z=8, int SHMEM_PAD=1>
__global__ void fusedHessianVesselnessKernel(
    const float* __restrict__ smoothedInput,
    float* __restrict__ output,
    int nRows, int nCols, int nSlices, float sigmaSq,
    int vesselnessMethod, // enum
    bool bright,
    float inv2Alpha2, float inv2Beta2, float inv2Gamma2, float alphaMeijering,
    bool firstPass
) {
    constexpr int HALO = 2;
    __shared__ float shInput[TILE_Z+2*HALO][TILE_Y+2*HALO][TILE_X+2*HALO + SHMEM_PAD];

    int x = blockIdx.x * TILE_X + threadIdx.x;
    int y = blockIdx.y * TILE_Y + threadIdx.y;
    int z = blockIdx.z * TILE_Z + threadIdx.z;
    int sx = threadIdx.x + HALO;
    int sy = threadIdx.y + HALO;
    int sz = threadIdx.z + HALO;

    // Shared memory load
    for (int dz = threadIdx.z; dz < TILE_Z + 2*HALO; dz += blockDim.z)
    for (int dy = threadIdx.y; dy < TILE_Y + 2*HALO; dy += blockDim.y)
    for (int dx = threadIdx.x; dx < TILE_X + 2*HALO; dx += blockDim.x)
    {
        int gx = blockIdx.x * TILE_X + dx - HALO;
        int gy = blockIdx.y * TILE_Y + dy - HALO;
        int gz = blockIdx.z * TILE_Z + dz - HALO;
        int rx = reflectCoord(gx, nRows);
        int ry = reflectCoord(gy, nCols);
        int rz = reflectCoord(gz, nSlices);
        shInput[dz][dy][dx] = smoothedInput[linearIndex3D(rx, ry, rz, nRows, nCols)];
    }
    __syncthreads();

    // Main computation
    if (x < nRows && y < nCols && z < nSlices) {
        // Hessian as before
        float xm2 = shInput[sz][sy][sx-2], xm1 = shInput[sz][sy][sx-1], xc = shInput[sz][sy][sx], xp1 = shInput[sz][sy][sx+1], xp2 = shInput[sz][sy][sx+2];
        float dxx = fmaf(-1.0f, xm2, 0.f); dxx = fmaf( 16.0f, xm1, dxx); dxx = fmaf(-30.0f, xc, dxx); dxx = fmaf(16.0f, xp1, dxx); dxx = fmaf(-1.0f, xp2, dxx); dxx /= float(FINITE_DIFF_5PT_DIVISOR);

        float ym2 = shInput[sz][sy-2][sx], ym1 = shInput[sz][sy-1][sx], yc = shInput[sz][sy][sx], yp1 = shInput[sz][sy+1][sx], yp2 = shInput[sz][sy+2][sx];
        float dyy = fmaf(-1.0f, ym2, 0.f); dyy = fmaf(16.0f, ym1, dyy); dyy = fmaf(-30.0f, yc, dyy); dyy = fmaf(16.0f, yp1, dyy); dyy = fmaf(-1.0f, yp2, dyy); dyy /= float(FINITE_DIFF_5PT_DIVISOR);

        float zm2 = shInput[sz-2][sy][sx], zm1 = shInput[sz-1][sy][sx], zc = shInput[sz][sy][sx], zp1 = shInput[sz+1][sy][sx], zp2 = shInput[sz+2][sy][sx];
        float dzz = fmaf(-1.0f, zm2, 0.f); dzz = fmaf(16.0f, zm1, dzz); dzz = fmaf(-30.0f, zc, dzz); dzz = fmaf(16.0f, zp1, dzz); dzz = fmaf(-1.0f, zp2, dzz); dzz /= float(FINITE_DIFF_5PT_DIVISOR);

        float dxy = 0.25f * (shInput[sz][sy+1][sx+1] - shInput[sz][sy+1][sx-1] - shInput[sz][sy-1][sx+1] + shInput[sz][sy-1][sx-1]);
        float dxz = 0.25f * (shInput[sz+1][sy][sx+1] - shInput[sz-1][sy][sx+1] - shInput[sz+1][sy][sx-1] + shInput[sz-1][sy][sx-1]);
        float dyz = 0.25f * (shInput[sz+1][sy+1][sx] - shInput[sz-1][sy+1][sx] - shInput[sz+1][sy-1][sx] + shInput[sz-1][sy-1][sx]);

        if ((VesselnessMethod)vesselnessMethod == METHOD_MEIJERING) {
            float trace = dxx + dyy + dzz;
            float delta = alphaMeijering * trace;
            dxx += delta; dyy += delta; dzz += delta;
        }

        // Eigenvalues
        float A11 = dxx * sigmaSq, A22 = dyy * sigmaSq, A33 = dzz * sigmaSq, A12 = dxy * sigmaSq, A13 = dxz * sigmaSq, A23 = dyz * sigmaSq;
        float l1, l2, l3;
        computeSymmetricEigenvalues3x3(A11, A22, A33, A12, A13, A23, l1, l2, l3);

        // Vesselness selection
        float vesselness = 0.f;
        switch ((VesselnessMethod)vesselnessMethod) {
        case METHOD_FRANGI:
            vesselness = vesselnessFrangi(l1, l2, l3, bright, inv2Alpha2, inv2Beta2, inv2Gamma2); break;
        case METHOD_SATO:
            vesselness = vesselnessSato(l1, l2, l3, bright, inv2Alpha2, inv2Beta2); break;
        case METHOD_MEIJERING:
            vesselness = neuritenessMeijering(l1, l2, l3, bright); break;
        case METHOD_JERMAN:
            // Jerman mapping:
            // inv2Alpha2 → tau
            // inv2Beta2  → lambda3minGuess
            // inv2Gamma2 → structureSensitivity threshold
            vesselness = vesselnessJerman(l1, l2, l3, bright, inv2Alpha2, inv2Beta2, inv2Gamma2); break;
        }

        size_t idx = linearIndex3D(x, y, z, nRows, nCols);
        if (firstPass) output[idx] = vesselness;
        else           output[idx] = fmaxf(output[idx], vesselness);
    }
}

// ======= Launch Wrapper =======

void launchFusedHessianVesselnessKernel(
    const float* smoothedInput, float* output, int nRows, int nCols, int nSlices,
    cudaStream_t stream,
    float sigmaSq, int vesselnessMethod, bool bright,
    float inv2Alpha2, float inv2Beta2, float inv2Gamma2, float alphaMeijering, bool firstPass)
{
    constexpr int TX = 8, TY = 8, TZ = 8, SHMEM_PAD = 1;
    dim3 blockDim(TX, TY, TZ);
    dim3 gridDim((nRows + TX - 1) / TX, (nCols + TY - 1) / TY, (nSlices + TZ - 1) / TZ);

    fusedHessianVesselnessKernel<TX,TY,TZ,SHMEM_PAD>
        <<<gridDim, blockDim, 0, stream>>>(
            smoothedInput, output,
            nRows, nCols, nSlices,
            sigmaSq, vesselnessMethod, bright,
            inv2Alpha2, inv2Beta2, inv2Gamma2, alphaMeijering, firstPass
        );
    cudaCheck(cudaPeekAtLastError());
}

__global__ void maxReduceKernel(const float* input, float* out, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into shared memory
    sdata[tid] = (i < n) ? input[i] : -FLT_MAX;
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void normalize_and_project(
    float* outputDev, const float* buffer2, const float* buffer2Max, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float normVal = (*buffer2Max > 0.f) ? buffer2[idx] / *buffer2Max : 0.f;
        outputDev[idx] = fmaxf(outputDev[idx], normVal);
    }
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

    // Each array is half, each float is 4 bytes:
    const int maxKernelLen = prop.totalConstMem / (2 * sizeof(float)); // 2 arrays: gCoef, dCoef

    //--- Inputs ---
    const mxGPUArray* input = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(input) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(input) != 3)
        mexErrMsgIdAndTxt("fibermetric_gpu:type", "Input must be 3D single gpuArray.");
    const mwSize* dims = mxGPUGetDimensions(input);
    int nRows = int(dims[0]), nCols = int(dims[1]), nSlices = int(dims[2]);
    size_t total = (size_t)nRows * (size_t)nCols * (size_t)nSlices;
    int blocks = int((total + threadsPerBlock - 1) / threadsPerBlock);
    const float* inputDev = static_cast<const float*>(mxGPUGetDataReadOnly(input));

    double sigmaFrom = mxGetScalar(prhs[1]);
    double sigmaTo   = mxGetScalar(prhs[2]);
    double sigmaStep = mxGetScalar(prhs[3]);
    float alpha      = float(mxGetScalar(prhs[4]));
    float beta       = float(mxGetScalar(prhs[5]));
    float jermanTau  = alpha;
    float jermanC    = beta;
    double structureSensitivity = mxGetScalar(prhs[6]);
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

    int vesselnessMethod = METHOD_FRANGI;
    if (useFrangi) vesselnessMethod = METHOD_FRANGI;
    else if (useSato) vesselnessMethod = METHOD_SATO;
    else if (useMeijering) vesselnessMethod = METHOD_MEIJERING;
    else if (useJerman) vesselnessMethod = METHOD_JERMAN;

    //--- Workspace Buffers and Output Allocation ---
    mxGPUArray* output = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outputDev = static_cast<float*>(mxGPUGetData(output));
    float *buffer1, *buffer2, *maxTemp, *buffer2Max;
    size_t bytes = total * sizeof(float);
    cudaCheck(cudaMalloc(&buffer1, bytes));
    cudaCheck(cudaMalloc(&buffer2, bytes));
    cudaCheck(cudaMalloc(&maxTemp, blocks * sizeof(float)));
    cudaCheck(cudaMalloc(&buffer2Max, sizeof(float)));

    //--- Sigma List and Scale Norm ---
    std::vector<double> sigmaList;
    for (double s = sigmaFrom; s <= sigmaTo + 1e-7; s += sigmaStep)
        sigmaList.push_back(s);

    constexpr float alphaMeijering = -0.25f; // for 3D, can compute as needed
    const float inv2Alpha2 = static_cast<float>(1.0 / (2.0 * double(alpha) * double(alpha)));
    const float inv2Beta2  = static_cast<float>(1.0 / (2.0 * double(beta)  * double(beta )));
    const float gamma      = static_cast<float>(structureSensitivity);
    const float inv2Gamma2 = static_cast<float>(1.0 / (2.0 * structureSensitivity * structureSensitivity));

    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreate(&stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    //--- Main Loop Over Scales ---
    for (size_t sigmaIdx = 0; sigmaIdx < sigmaList.size(); ++sigmaIdx) {
        double sigma = sigmaList[sigmaIdx];
        double sigmaSq = sigma * sigma;

        // 1. Create Gaussian and derivative kernels for this sigma
        float* gaussKernelDev = nullptr;
        int halfWidth = int(std::ceil(GAUSS_HALFWIDTH_MULT * sigma));
        int kernelLen = 2 * halfWidth + 1;

        int maxTileSize = (prop.sharedMemPerBlock / sizeof(float)) - (kernelLen - 1);
        if (maxTileSize < 16) maxTileSize = 16;
        int tileSize = std::min(DEFAULT_TILE_SIZE, maxTileSize);

        buildGaussianKernelDevice(gaussKernelDev, kernelLen, sigma, sigmaSq, threadsPerBlock);

        // 2. Upload kernels to constant memory if eligible
        bool useConstMem = (kernelLen <= maxKernelLen) && (kernelLen <= MAX_CONST_COEF_LEN);
        if (useConstMem) {
            std::vector<float> gaussHost(kernelLen);
            cudaCheck(cudaMemcpy(gaussHost.data(), gaussKernelDev, kernelLen * sizeof(float), cudaMemcpyDeviceToHost));
            uploadCoefficientsToConstantMemory(gaussHost.data(), kernelLen);
        }

        // 3. Clean up any old graph (always recapture per sigma for correctness) ---
        if (graphExec) { cudaCheck(cudaGraphExecDestroy(graphExec)); graphExec = nullptr; }
        if (graph)     { cudaCheck(cudaGraphDestroy(graph));         graph     = nullptr; }

        // ================================
        // --- All smoothing and vesselness calculation before capture ---
        // ================================

        // 4. Gaussian smoothing (separable, X, Y, Z)
        launchSeparableConvolutionDevice(0, inputDev, buffer1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream, useConstMem, tileSize);
        launchSeparableConvolutionDevice(1, buffer1 , buffer2, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream, useConstMem, tileSize);
        launchSeparableConvolutionDevice(2, buffer2 , buffer1, gaussKernelDev, kernelLen, nRows, nCols, nSlices, threadsPerBlock, stream, useConstMem, tileSize);

        // 5. Hessian diagonals, Scale normalization, Eigenvalue decomposition, Vesselness all at traverse
        if (vesselnessMethod != METHOD_JERMAN) {
            bool firstPass = (sigmaIdx == 0);
            // --- Everything can be inside the graph for Frangi, Sato, Meijering ---
            cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launchFusedHessianVesselnessKernel(
                buffer1, outputDev, nRows, nCols, nSlices, stream, float(sigmaSq),
                vesselnessMethod, bright, inv2Alpha2, inv2Beta2, inv2Gamma2,
                alphaMeijering, firstPass
            );
            cudaCheck(cudaStreamEndCapture(stream, &graph));
            cudaCheck(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
            cudaCheck(cudaGraphLaunch(graphExec, stream));
        } else {
            // --- Jerman: max reduction is outside graph! ---
            float lambda3minGuess = float(-double(jermanC) * sigmaSq);

            // (1) Write vesselness to buffer2 (not inside graph)
            launchFusedHessianVesselnessKernel(
                buffer1, buffer2, nRows, nCols, nSlices, stream, float(sigmaSq),
                vesselnessMethod, bright,  jermanTau, lambda3minGuess, structureSensitivity,
                alphaMeijering, true
            );

            // (2) Compute max reduction outside of capture!
            maxReduceKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(buffer2, maxTemp, total);
            if (blocks > 1)
                maxReduceKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(maxTemp, buffer2Max, blocks);
            else
                cudaMemcpyAsync(buffer2Max, maxTemp, sizeof(float), cudaMemcpyDeviceToDevice, stream);

            // (3) Now only normalization/projection inside graph
            cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            normalize_and_project<<<blocks, threadsPerBlock, 0, stream>>>(
                outputDev, buffer2, buffer2Max, total
            );
            cudaCheck(cudaPeekAtLastError());
            cudaCheck(cudaStreamEndCapture(stream, &graph));
            cudaCheck(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
            cudaCheck(cudaGraphLaunch(graphExec, stream));
        }
        cudaFree(gaussKernelDev);
    }

    // Synchronize & Cleanup
    cudaCheck(cudaStreamSynchronize(stream));
    if (graphExec) cudaCheck(cudaGraphExecDestroy(graphExec));
    if (graph)     cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));

    //--- Output ---
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);

    //--- Free device buffers ---
    cudaFree(buffer1); cudaFree(buffer2); cudaFree(maxTemp); cudaFree(buffer2Max);
    mxGPUDestroyGPUArray(input); mxGPUDestroyGPUArray(output);
}
