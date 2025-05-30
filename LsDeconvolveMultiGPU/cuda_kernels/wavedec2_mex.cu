// ===============================
// wavedec2_mex.cu (gpuArray + debug)
// ===============================
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "wavelet_filters.cuh"
#include "cuda_kernels/convolve2d_downsample.cuh"
#include <string>

#define CHECK_CUDA(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        mexErrMsgIdAndTxt("CUDA:Runtime", "%s: %s", msg, cudaGetErrorString(err)); \
    } \
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    mxInitGPU();

    if (nrhs != 3)
        mexErrMsgTxt("Usage: wavedec2_mex(img, levels, waveletName)");

    // Get input image (gpuArray)
    if (!mxIsGPUArray(prhs[0]))
        mexErrMsgTxt("First argument must be a gpuArray image.");
    const mxGPUArray* img_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const float* d_img = static_cast<const float*>(mxGPUGetDataReadOnly(img_gpu));
    const mwSize* dims = mxGPUGetDimensions(img_gpu);
    int height = static_cast<int>(dims[0]);
    int width  = static_cast<int>(dims[1]);
    mexPrintf("Image size: %d x %d\n", height, width);

    // Get decomposition levels
    int levels = static_cast<int>(mxGetScalar(prhs[1]));
    mexPrintf("Decomposition levels: %d\n", levels);

    // Get wavelet name
    char waveletName[32];
    mxGetString(prhs[2], waveletName, sizeof(waveletName));
    std::string wname(waveletName);
    mexPrintf("Wavelet: %s\n", waveletName);

    // Load filters
    std::vector<float> Lo_D, Hi_D;
    if (!get_analysis_filters(wname, Lo_D, Hi_D))
        mexErrMsgTxt("Unsupported wavelet.");
    mexPrintf("Filter length: %d\n", (int)Lo_D.size());

    // Allocate storage
    std::vector<int2> S_host(levels + 1);
    std::vector<float*> subbands;
    float* d_C = nullptr;

    // Perform decomposition
    mexPrintf("Starting wavelet decomposition...\n");

    size_t total_coeffs = perform_wavelet_decomposition(
        d_img, width, height, levels,
        Lo_D.data(), Lo_D.size(),
        Hi_D.data(), Hi_D.size(),
        d_C, S_host, subbands
    );
    CHECK_CUDA("Wavelet decomposition");

    mexPrintf("Total coefficients: %zu\n", total_coeffs);

    // Output gpuArray C
    mwSize c_dims[1] = {static_cast<mwSize>(total_coeffs)};
    mxGPUArray* C_gpu = mxGPUCreateGPUArray(1, c_dims, mxSINGLE_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    float* d_outC = static_cast<float*>(mxGPUGetData(C_gpu));
    cudaMemcpy(d_outC, d_C, sizeof(float) * total_coeffs, cudaMemcpyDeviceToDevice);
    CHECK_CUDA("Copy C to output");

    // Output gpuArray S (converted to int2)
    mwSize s_dims[1] = {static_cast<mwSize>(S_host.size())};
    mxGPUArray* S_gpu = mxGPUCreateGPUArray(1, s_dims, mxUINT32_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int2* d_outS = static_cast<int2*>(mxGPUGetData(S_gpu));
    cudaMemcpy(d_outS, S_host.data(), sizeof(int2) * S_host.size(), cudaMemcpyHostToDevice);
    CHECK_CUDA("Copy S to output");

    // Set outputs
    plhs[0] = mxGPUCreateMxArrayOnGPU(C_gpu);
    plhs[1] = mxGPUCreateMxArrayOnGPU(S_gpu);

    // Cleanup
    cudaFree(d_C);
    mxGPUDestroyGPUArray(C_gpu);
    mxGPUDestroyGPUArray(S_gpu);
    mxGPUDestroyGPUArray(img_gpu);

    mexPrintf("Finished wavedec2_mex.\n");
}
