// gauss3d_mex.cu - 3D Gaussian filtering for MATLAB GPU, kernel warning via argument
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#define MAX_KERNEL_SIZE 255

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        mexErrMsgIdAndTxt("gauss3d:cuda", "CUDA error: %s", cudaGetErrorString(err)); \
} while(0)

template <typename T>
__global__ void gauss1d_kernel(
    const T* src, T* dst, int nx, int ny, int nz,
    const float* kernel, int klen, int dim
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int size[3] = {nx, ny, nz};
    int center = klen / 2;

    int idx = iz * nx * ny + iy * nx + ix;
    T val = 0;

    for (int k = 0; k < klen; ++k) {
        int offset = k - center;
        int ci = (dim == 0) ? ix + offset : (dim == 1) ? iy + offset : iz + offset;
        ci = min(max(ci, 0), size[dim] - 1); // replicate boundary
        int cidx;
        if (dim == 0)
            cidx = iz * nx * ny + iy * nx + ci;
        else if (dim == 1)
            cidx = iz * nx * ny + ci * nx + ix;
        else
            cidx = ci * nx * ny + iy * nx + ix;
        val += src[cidx] * kernel[k];
    }
    dst[idx] = val;
}

void make_gaussian_kernel(float sigma, int ksize, float* kernel, int* klen) {
    int half = (ksize - 1) / 2;
    *klen = ksize;
    float sum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        float v = expf(-0.5f * (i * i) / (sigma * sigma));
        kernel[i + half] = v;
        sum += v;
    }
    // Normalize kernel
    for (int i = 0; i < ksize; ++i)
        kernel[i] /= sum;
}

int recommended_kernel_size(float sigma) {
    return 2 * (int)ceilf(3.0f * sigma) + 1; // Conventional "full coverage"
}

template<typename T>
void run_gauss3d(T* bufA, T* bufB, int nx, int ny, int nz, float sigma[3], int ksize[3], bool warn_ksize) {
    float *h_kernel = new float[MAX_KERNEL_SIZE];
    float *d_kernel;
    int klen;

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    T *src = bufA;
    T *dst = bufB;
    for (int dim = 0; dim < 3; ++dim) {
        make_gaussian_kernel(sigma[dim], ksize[dim], h_kernel, &klen);
        if (warn_ksize) {
            int kmin = recommended_kernel_size(sigma[dim]);
            if (ksize[dim] < kmin) {
                mexWarnMsgIdAndTxt("gauss3d:kernel",
                  "Kernel size for axis %d (%d) is too small for sigma=%.3f (recommended at least %d). Results may be inaccurate.\n"
                  "To disable this warning, call gauss3d_mex(..., false).", dim+1, ksize[dim], sigma[dim], kmin);
            }
        }
        CUDA_CHECK(cudaMalloc(&d_kernel, klen * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, klen * sizeof(float), cudaMemcpyHostToDevice));

        gauss1d_kernel<T><<<grid, block>>>(src, dst, nx, ny, nz, d_kernel, klen, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));

        // Swap pointers for next pass
        T* tmp = src;
        src = dst;
        dst = tmp;
    }
    delete[] h_kernel;
    // Output is in src after 3 passes
}

void parse_sigma_ksize(const mxArray* sigma_in, const mxArray* ksize_in, float sigma[3], int ksize[3], const mwSize* sz) {
    // sigma_in: scalar or length-3
    if (mxIsScalar(sigma_in)) {
        float v = (float)mxGetScalar(sigma_in);
        sigma[0] = sigma[1] = sigma[2] = v;
    } else if (mxGetNumberOfElements(sigma_in) == 3) {
        double* ps = mxGetPr(sigma_in);
        for (int i=0;i<3;++i) sigma[i] = (float)ps[i];
    } else {
        mexErrMsgIdAndTxt("gauss3d:sigma", "Sigma must be scalar or length-3 vector.");
    }

    // ksize_in: not provided (auto), scalar, or length-3
    if (ksize_in == NULL) {
        // auto: recommended size for each axis
        for (int i=0;i<3;++i)
            ksize[i] = recommended_kernel_size(sigma[i]);
    } else if (mxIsScalar(ksize_in)) {
        int k = (int)mxGetScalar(ksize_in);
        ksize[0]=ksize[1]=ksize[2]=k;
    } else if (mxGetNumberOfElements(ksize_in)==3) {
        double* pk = mxGetPr(ksize_in);
        for (int i=0;i<3;++i) ksize[i]=(int)pk[i];
    } else {
        mexErrMsgIdAndTxt("gauss3d:ksize", "Kernel size must be empty, scalar or length-3 vector.");
    }
}

// MEX entry
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs < 2)
        mexErrMsgIdAndTxt("gauss3d:nrhs", "Need at least input array and sigma");

    if (!mxIsGPUArray(prhs[0]))
        mexErrMsgIdAndTxt("gauss3d:input", "Input must be a gpuArray.");

    mxGPUArray const* img_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* sz = mxGPUGetDimensions(img_gpu);
    int nd = mxGPUGetNumberOfDimensions(img_gpu);
    if (nd != 3)
        mexErrMsgIdAndTxt("gauss3d:ndims", "Input must be 3D.");

    int nx = (int)sz[0], ny = (int)sz[1], nz = (int)sz[2];

    float sigma[3];
    int ksize[3];
    const mxArray* ksize_in = (nrhs >= 3) ? prhs[2] : NULL;
    parse_sigma_ksize(prhs[1], ksize_in, sigma, ksize, sz);

    // Warn on kernel size? Default is false
    bool warn_ksize = false;
    if (nrhs >= 4) {
        if (mxIsLogicalScalar(prhs[3]))
            warn_ksize = mxIsLogicalScalarTrue(prhs[3]);
        else if (mxIsDouble(prhs[3]))
            warn_ksize = (mxGetScalar(prhs[3]) != 0);
    }

    mxClassID cls = mxGPUGetClassID(img_gpu);
    mxGPUArray *A_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *B_gpu = mxGPUCreateGPUArray(3, sz, cls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Copy input to A_gpu
    size_t N = (size_t)nx*ny*nz;
    size_t elsize = (cls == mxSINGLE_CLASS) ? sizeof(float) : sizeof(double);
    void* input_ptr = (void*)mxGPUGetDataReadOnly(img_gpu);
    void* A_ptr = (void*)mxGPUGetData(A_gpu);
    CUDA_CHECK(cudaMemcpy(A_ptr, input_ptr, N * elsize, cudaMemcpyDeviceToDevice));
    void* B_ptr = (void*)mxGPUGetData(B_gpu);

    // Run filter (output may be in A or B)
    if (cls == mxSINGLE_CLASS)
        run_gauss3d<float>((float*)A_ptr, (float*)B_ptr, nx, ny, nz, sigma, ksize, warn_ksize);
    else if (cls == mxDOUBLE_CLASS)
        run_gauss3d<double>((double*)A_ptr, (double*)B_ptr, nx, ny, nz, sigma, ksize, warn_ksize);
    else
        mexErrMsgIdAndTxt("gauss3d:class", "Input must be single or double");

    // Output is always in src (A_gpu) after last swap if passes is odd (3)
    plhs[0] = mxGPUCreateMxArrayOnGPU(A_gpu);

    mxGPUDestroyGPUArray(img_gpu);
    mxGPUDestroyGPUArray(A_gpu);
    mxGPUDestroyGPUArray(B_gpu);
}
