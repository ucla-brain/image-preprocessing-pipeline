// ===============================
// waverec2_mex.cu (gpuArray-compatible)
// ===============================
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "wavelet_filters.cuh"
#include "cuda_kernels/convolve2d_upsample.cuh"
#include <string>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Initialize GPU API
    mxInitGPU();

    if (nrhs != 3)
        mexErrMsgTxt("Usage: waverec2_mex(C, S, waveletName)");

    // Retrieve input C (gpuArray)
    if (!mxIsGPUArray(prhs[0]))
        mexErrMsgTxt("C must be a gpuArray");
    const mxGPUArray* C_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const float* d_C = static_cast<const float*>(mxGPUGetDataReadOnly(C_gpu));

    // Retrieve input S (gpuArray)
    if (!mxIsGPUArray(prhs[1]))
        mexErrMsgTxt("S must be a gpuArray");
    const mxGPUArray* S_gpu = mxGPUCreateFromMxArray(prhs[1]);
    const int2* d_S = static_cast<const int2*>(mxGPUGetDataReadOnly(S_gpu));

    // Retrieve wavelet name
    char waveletName[32];
    mxGetString(prhs[2], waveletName, sizeof(waveletName));
    std::string wname(waveletName);

    // Get synthesis filters
    std::vector<float> Lo_R, Hi_R;
    if (!get_synthesis_filters(wname, Lo_R, Hi_R))
        mexErrMsgTxt("Unsupported wavelet");

    // Allocate and run inverse wavelet transform
    int levels = mxGPUGetNumberOfElements(S_gpu) - 1;
    int2 imgSize = d_S[0]; // final reconstructed size

    float* d_out;
    cudaMalloc(&d_out, sizeof(float) * imgSize.x * imgSize.y);

    reconstruct_full_image(d_C, d_S, levels,
                           Lo_R.data(), Lo_R.size(),
                           Hi_R.data(), Hi_R.size(),
                           d_out);

    // Create output mxGPUArray
    mwSize dims[2] = {static_cast<mwSize>(imgSize.x), static_cast<mwSize>(imgSize.y)};
    mxGPUArray* out_gpu = mxGPUCreateGPUArray(2, dims, mxSINGLE_CLASS, mxREAL,
                                              MX_GPU_DO_NOT_INITIALIZE);
    float* outData = static_cast<float*>(mxGPUGetData(out_gpu));
    cudaMemcpy(outData, d_out, sizeof(float) * imgSize.x * imgSize.y, cudaMemcpyDeviceToDevice);

    plhs[0] = mxGPUCreateMxArrayOnGPU(out_gpu);

    // Cleanup
    cudaFree(d_out);
    mxGPUDestroyGPUArray(out_gpu);
    mxGPUDestroyGPUArray(C_gpu);
    mxGPUDestroyGPUArray(S_gpu);
}
