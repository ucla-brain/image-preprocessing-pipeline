/*==============================================================================
  if_else.cu

  Fast, vectorized, CPU and GPU-accelerated elementwise conditional selection
  for MATLAB arrays (MEX). Fully compatible with gpuArray and CPU arrays.

  ------------------------------------------------------------------------------
  Author:    Keivan Moradi (2025) [with ChatGPT-4o assistance]
  License:   GPL v3
  ------------------------------------------------------------------------------

  OVERVIEW:
  ---------
    This MEX function implements a fast, broadcast-compatible elementwise
    conditional (if-else) operator, similar to R's if_else() or numpy.where(),
    for both CPU and GPU (gpuArray) arrays in MATLAB.

    For each element:
        out = cond ? x : y

    The implementation uses:
      - AVX2 SIMD and OpenMP on CPU for maximum throughput
      - CUDA on GPU for high parallel performance
      - Scalar fallback if AVX2/OpenMP is unavailable

  USAGE:
  ------
    out = if_else(cond, x, y)

    Inputs:
      cond : Logical array or gpuArray (any shape)
      x    : Numeric array or gpuArray (single or double, same shape as cond)
      y    : Numeric array or gpuArray (single or double, same shape as cond)

    Output:
      out  : Numeric array (same type and shape as x/y; returns gpuArray if any input is gpuArray)

    - All inputs must be the same size (no broadcasting)
    - Both x and y must have the same numeric type (single or double)

  FEATURES:
  ---------
    - **Fast CPU execution:** Uses AVX2 vector instructions and OpenMP parallelism if available
    - **GPU acceleration:** Uses custom CUDA kernel for optimal throughput on supported GPUs
    - **Drop-in replacement:** Much faster than MATLAB arrayfun/logical indexing for large arrays
    - **Robust:** Handles all dimensions/shapes supported by MATLAB, including edge cases

  EXAMPLES:
  ---------
    % CPU arrays
    cond = rand(100000,1) > 0.5;
    x = randn(100000,1,'single');
    y = zeros(100000,1,'single');
    out = if_else(cond, x, y);

    % GPU arrays
    cond = gpuArray(rand(100000,1) > 0.5);
    x = gpuArray.ones(100000,1,'single');
    y = gpuArray.zeros(100000,1,'single');
    out = if_else(cond, x, y);  % returns gpuArray

  PERFORMANCE NOTES:
  ------------------
    - For large arrays, outperforms MATLAB's logical indexing, arrayfun, and arithmetic alternatives
    - On CPU, leverages AVX2 SIMD (if available) and OpenMP threading
    - On GPU, launches one thread per element
    - Falls back to scalar C++ if AVX2/OpenMP not supported

  LIMITATIONS:
  -----------
    - Only supports 'single' and 'double' numeric types for x and y (extendable)
    - No broadcasting: all arrays must be the same size and shape
    - Not intended for complex, integer, or struct types (but can be extended)
    - On very small arrays, MATLAB's logical indexing may be equally fast

==============================================================================*/

#include "mex.h"
#include <stdint.h>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#endif

#include <immintrin.h>

// ===================== SIMD CPU path (float) ===========================
void if_else_cpu_simd(const bool* cond, const float* x, const float* y, float* out, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const size_t vec_width = 8;
    for (; i + vec_width - 1 < n; i += vec_width) {
        __m256 msk = _mm256_castsi256_ps(
            _mm256_set_epi32(
                cond[i+7] ? 0xFFFFFFFF : 0,
                cond[i+6] ? 0xFFFFFFFF : 0,
                cond[i+5] ? 0xFFFFFFFF : 0,
                cond[i+4] ? 0xFFFFFFFF : 0,
                cond[i+3] ? 0xFFFFFFFF : 0,
                cond[i+2] ? 0xFFFFFFFF : 0,
                cond[i+1] ? 0xFFFFFFFF : 0,
                cond[i+0] ? 0xFFFFFFFF : 0
            )
        );
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vout = _mm256_blendv_ps(vy, vx, msk);
        _mm256_storeu_ps(out + i, vout);
    }
#endif
    #pragma omp parallel for if (n > 10000)
    for (ptrdiff_t j = i; j < static_cast<ptrdiff_t>(n); ++j)
        out[j] = cond[j] ? x[j] : y[j];
}

// ===================== SIMD CPU path (double) ===========================
void if_else_cpu_simd(const bool* cond, const double* x, const double* y, double* out, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const size_t vec_width = 4;
    for (; i + vec_width - 1 < n; i += vec_width) {
        __m256d msk = _mm256_castsi256_pd(
            _mm256_set_epi64x(
                cond[i+3] ? 0xFFFFFFFFFFFFFFFFull : 0,
                cond[i+2] ? 0xFFFFFFFFFFFFFFFFull : 0,
                cond[i+1] ? 0xFFFFFFFFFFFFFFFFull : 0,
                cond[i+0] ? 0xFFFFFFFFFFFFFFFFull : 0
            )
        );
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
        __m256d vout = _mm256_blendv_pd(vy, vx, msk);
        _mm256_storeu_pd(out + i, vout);
    }
#endif
    #pragma omp parallel for if (n > 10000)
    for (ptrdiff_t j = i; j < static_cast<ptrdiff_t>(n); ++j)
        out[j] = cond[j] ? x[j] : y[j];
}

// =========================== CUDA GPU path ===============================
#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
template<typename T>
__global__ void if_else_kernel(const bool* cond, const T* x, const T* y, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = cond[idx] ? x[idx] : y[idx];
}

template<typename T>
void if_else_gpu(const mxArray* cond_in, const mxArray* x_in, const mxArray* y_in, mxArray** out_out) {
    const mxGPUArray *cond, *x, *y;
    mxGPUArray *out;
    const bool* d_cond;
    const T* d_x; const T* d_y; T* d_out;
    size_t n;

    cond = mxGPUCreateFromMxArray(cond_in);
    x    = mxGPUCreateFromMxArray(x_in);
    y    = mxGPUCreateFromMxArray(y_in);
    n = mxGPUGetNumberOfElements(x);

    d_cond = static_cast<const bool*>(mxGPUGetDataReadOnly(cond));
    d_x    = static_cast<const T*>(mxGPUGetDataReadOnly(x));
    d_y    = static_cast<const T*>(mxGPUGetDataReadOnly(y));

    out = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(x), mxGPUGetDimensions(x),
                              mxGPUGetClassID(x), mxGPUGetComplexity(x), MX_GPU_DO_NOT_INITIALIZE);
    d_out = static_cast<T*>(mxGPUGetData(out));

    int blockSize = 256;
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    if_else_kernel<T><<<gridSize, blockSize>>>(d_cond, d_x, d_y, d_out, n);
    cudaDeviceSynchronize();

    *out_out = mxGPUCreateMxArrayOnGPU(out);

    mxGPUDestroyGPUArray(cond);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);
    mxGPUDestroyGPUArray(out);
}
#endif

// ======================= Main dispatch: type-safe =======================

void do_if_else(const mxArray* cond_in, const mxArray* x_in, const mxArray* y_in, mxArray** out_out, bool isGPU) {
#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
    if (isGPU) {
        mxClassID class_id;
        {
            mxGPUArray const* garr = mxGPUCreateFromMxArray(x_in);
            class_id = mxGPUGetClassID(garr);
            mxGPUDestroyGPUArray(garr);
        }
        if (class_id == mxSINGLE_CLASS)
            if_else_gpu<float>(cond_in, x_in, y_in, out_out);
        else if (class_id == mxDOUBLE_CLASS)
            if_else_gpu<double>(cond_in, x_in, y_in, out_out);
        else
            mexErrMsgTxt("Only single/double supported for GPU arrays.");
        return;
    }
#endif

    size_t n = mxGetNumberOfElements(x_in);
    mxClassID class_id = mxGetClassID(x_in);

    if (class_id == mxSINGLE_CLASS) {
        const float* x = static_cast<const float*>(mxGetData(x_in));
        const float* y = static_cast<const float*>(mxGetData(y_in));
        const bool* cond = static_cast<const bool*>(mxGetData(cond_in));
        *out_out = mxCreateNumericArray(mxGetNumberOfDimensions(x_in), mxGetDimensions(x_in), class_id, mxREAL);
        float* out = static_cast<float*>(mxGetData(*out_out));
        if_else_cpu_simd(cond, x, y, out, n);
    } else if (class_id == mxDOUBLE_CLASS) {
        const double* x = static_cast<const double*>(mxGetData(x_in));
        const double* y = static_cast<const double*>(mxGetData(y_in));
        const bool* cond = static_cast<const bool*>(mxGetData(cond_in));
        *out_out = mxCreateNumericArray(mxGetNumberOfDimensions(x_in), mxGetDimensions(x_in), class_id, mxREAL);
        double* out = static_cast<double*>(mxGetData(*out_out));
        if_else_cpu_simd(cond, x, y, out, n);
    } else {
        mexErrMsgTxt("Only single/double supported for CPU arrays.");
    }
}

bool is_gpu_array(const mxArray* arr) {
#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
    return mxIsGPUArray(arr);
#else
    return false;
#endif
}

// ============================= Entry Point ==============================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 3)
        mexErrMsgTxt("Usage: out = if_else(cond, x, y);");

#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
    bool useGPU = is_gpu_array(prhs[0]) || is_gpu_array(prhs[1]) || is_gpu_array(prhs[2]);
    if (useGPU)
        mxInitGPU();
#else
    bool useGPU = false;
#endif

    // Ensure input types match
    mxClassID class_id1, class_id2;
#if defined(__CUDACC__) || defined(MATLAB_MEXCMD_RELEASE)
    if (useGPU) {
        mxGPUArray const* garr1 = mxGPUCreateFromMxArray(prhs[1]);
        mxGPUArray const* garr2 = mxGPUCreateFromMxArray(prhs[2]);
        class_id1 = mxGPUGetClassID(garr1);
        class_id2 = mxGPUGetClassID(garr2);
        mxGPUDestroyGPUArray(garr1);
        mxGPUDestroyGPUArray(garr2);
    } else {
        class_id1 = mxGetClassID(prhs[1]);
        class_id2 = mxGetClassID(prhs[2]);
    }
#else
    class_id1 = mxGetClassID(prhs[1]);
    class_id2 = mxGetClassID(prhs[2]);
#endif
    if (class_id1 != class_id2)
        mexErrMsgTxt("x and y must have the same type (single or double).");

    do_if_else(prhs[0], prhs[1], prhs[2], &plhs[0], useGPU);
}
