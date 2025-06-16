/*
 * conv3d_mex.cu
 *
 * 3D Convolution with Replicate Boundary Conditions (GPU-Accelerated)
 * --------------------------------------------------------------------
 * This MEX function performs 3D convolution between an input volume and
 * a 3D kernel using GPU acceleration. It uses replicate (clamped) boundary
 * conditions, where out-of-bounds accesses are clamped to the edge.
 *
 * INPUTS (via MATLAB):
 *   - img   : 3D single-precision gpuArray (size: [X, Y, Z])
 *   - kernel: 3D single-precision gpuArray (size: [kx, ky, kz])
 *
 * OUTPUT:
 *   - out   : 3D single-precision gpuArray of same size as input image
 *
 * NOTES:
 *   - Input and kernel must be 3D `gpuArray` of class `single`.
 *   - The kernel is flipped in all three dimensions, as per convolution.
 *   - The output has the same size as the input (`same` convolution).
 *   - Replicate boundary conditions are used (edge clamping).
 *   - The function automatically handles memory cleanup using RAII.
 *   - CUDA kernel uses (8, 8, 4) block size for reasonable GPU occupancy.
 *   - This function is intended to be called from MATLAB as:
 *
 *       >> out = conv3d_mex(img, kernel);
 *
 * AUTHOR:
 *   Adapted and safety-patched by ChatGPT for Keivan Moradi (2025)
 *
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
        mexErrMsgIdAndTxt("conv3d_mex:CUDA", "CUDA error %s:%d: %s", file, line, cudaGetErrorString(code));
}

// RAII wrapper for mxGPUArray*
struct GpuHandle {
    mxGPUArray* ptr = nullptr;
    GpuHandle() = default;
    explicit GpuHandle(mxGPUArray* p) : ptr(p) {}
    ~GpuHandle() { if (ptr) mxGPUDestroyGPUArray(ptr); }
    // allow assignment like: handle = mxGPUCreate...
    GpuHandle& operator=(mxGPUArray* p) {
        if (ptr) mxGPUDestroyGPUArray(ptr);
        ptr = p;
        return *this;
    }
    // implicit conversion to mxGPUArray*
    operator mxGPUArray*() const { return ptr; }
};

__global__ void conv3d_kernel(
    const float* __restrict__ img, const float* __restrict__ kernel, float* __restrict__ out,
    int nx, int ny, int nz, int kx, int ky, int kz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int kx2 = kx / 2, ky2 = ky / 2, kz2 = kz / 2;

    float acc = 0.0f;
    #pragma unroll 1
    for (int dz = 0; dz < kz; dz++) {
        int iz = z + dz - kz2;
        iz = max(0, min(iz, nz - 1)); // replicate
        #pragma unroll 1
        for (int dy = 0; dy < ky; dy++) {
            int iy = y + dy - ky2;
            iy = max(0, min(iy, ny - 1));
            #pragma unroll 1
            for (int dx = 0; dx < kx; dx++) {
                int ix = x + dx - kx2;
                ix = max(0, min(ix, nx - 1));
                int fx = kx - 1 - dx, fy = ky - 1 - dy, fz = kz - 1 - dz;
                acc += img[ix + iy * nx + iz * nx * ny] *
                       kernel[fx + fy * kx + fz * kx * ky];
            }
        }
    }
    out[x + y * nx + z * nx * ny] = acc;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxInitGPU();

    if (nrhs != 2)
        mexErrMsgIdAndTxt("conv3d_mex:Args", "Requires two inputs: img, kernel (both gpuArray single 3D).");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("conv3d_mex:Args", "One output only.");

    // Validate types *before* allocating handles
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]))
        mexErrMsgIdAndTxt("conv3d_mex:Input", "Inputs must be gpuArray.");
    if (mxGPUGetClassID(mxGPUCreateFromMxArray(prhs[0])) != mxSINGLE_CLASS ||
        mxGPUGetNumberOfDimensions(mxGPUCreateFromMxArray(prhs[0])) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Input", "Image must be 3D gpuArray single.");
    if (mxGPUGetClassID(mxGPUCreateFromMxArray(prhs[1])) != mxSINGLE_CLASS ||
        mxGPUGetNumberOfDimensions(mxGPUCreateFromMxArray(prhs[1])) != 3)
        mexErrMsgIdAndTxt("conv3d_mex:Kernel", "Kernel must be 3D gpuArray single.");

    // Create RAII-managed GPU handles
    GpuHandle img(mxGPUCreateFromMxArray(prhs[0]));
    GpuHandle ker(mxGPUCreateFromMxArray(prhs[1]));
    GpuHandle out;

    const mwSize *isz = mxGPUGetDimensions(img);
    const mwSize *ksz = mxGPUGetDimensions(ker);

    int nx = static_cast<int>(isz[0]), ny = static_cast<int>(isz[1]), nz = static_cast<int>(isz[2]);
    int kx = static_cast<int>(ksz[0]), ky = static_cast<int>(ksz[1]), kz = static_cast<int>(ksz[2]);

    const float *d_img = static_cast<const float*>(mxGPUGetDataReadOnly(img));
    const float *d_ker = static_cast<const float*>(mxGPUGetDataReadOnly(ker));

    out = mxGPUCreateGPUArray(3, isz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float *d_out = static_cast<float*>(mxGPUGetData(out));

    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (nz + block.z - 1) / block.z);

    conv3d_kernel<<<grid, block>>>(d_img, d_ker, d_out, nx, ny, nz, kx, ky, kz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    out.ptr = nullptr; // Prevent RAII from destroying the returned data
}
