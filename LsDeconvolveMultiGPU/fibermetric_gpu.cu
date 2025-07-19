// vesselness_mex.cu -- MATLAB MEX (CUDA) for 3D Frangi/Vesselness filter (single-precision, all-in-one)
// Compile with: mexcuda -largeArrayDims vesselness_mex.cu
#include "mex.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

#define CHECK_CUDA(call) \
    do { cudaError_t err = (call); if (err != cudaSuccess) { \
        std::ostringstream oss; \
        oss << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        mexErrMsgIdAndTxt("vesselness:cuda", "%s", oss.str().c_str()); \
    }} while (0)

// ======================= UTILITY: 3D Indexing ========================
inline __host__ __device__ int sub2ind(int x, int y, int z, int sx, int sy, int sz) {
    return x + y*sx + z*sx*sy;
}

// ======================= CUDA KERNELS ================================

__global__ void cov3_kernel(
    const float* src, float* dst, const float* kernel,
    int sx, int sy, int sz, int kx, int ky, int kz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = sx * sy * sz;
    if (idx >= N) return;
    int x = idx % sx, y = (idx / sx) % sy, z = idx / (sx*sy);
    float sum = 0.0f, ksum = 0.0f;

    int hx = kx / 2, hy = ky / 2, hz = kz / 2;
    for (int dz = -hz; dz <= hz; ++dz)
        for (int dy = -hy; dy <= hy; ++dy)
            for (int dx = -hx; dx <= hx; ++dx) {
                int xx = x + dx, yy = y + dy, zz = z + dz;
                if (xx < 0 || xx >= sx || yy < 0 || yy >= sy || zz < 0 || zz >= sz) continue;
                int kidx = (dx+hx) + (dy+hy)*kx + (dz+hz)*kx*ky;
                int sidx = xx + yy*sx + zz*sx*sy;
                sum += src[sidx] * kernel[kidx];
                ksum += kernel[kidx];
            }
    dst[idx] = (ksum > 0.0f) ? sum / ksum : 0.0f;
}

__global__ void multiply_kernel(const float* src, float* dst, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx] * scale;
}

// --- Vesselness Kernel ---

#define GET(src, i, sx, sy, sz, ox, oy, oz) \
    src[(i) + (ox) + (oy)*(sx) + (oz)*(sx)*(sy)]

__global__ void vesselness_kernel(
    const float* src, float* dst, int sx, int sy, int sz,
    float alpha, float beta, float gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = sx*sy*sz;
    if (i >= N) return;

    int ix = i % sx, iy = (i / sx) % sy, iz = i / (sx*sy);
    if(ix<=0 || ix>=sx-1 || iy<=0 || iy>=sy-1 || iz<=0 || iz>=sz-1) return;

    float im_dx2 = -2.0f * GET(src,i,sx,sy,sz,0,0,0)
                   + GET(src,i,sx,sy,sz,-1,0,0)
                   + GET(src,i,sx,sy,sz,+1,0,0);
    float im_dy2 = -2.0f * GET(src,i,sx,sy,sz,0,0,0)
                   + GET(src,i,sx,sy,sz,0,-1,0)
                   + GET(src,i,sx,sy,sz,0,+1,0);
    float im_dz2 = -2.0f * GET(src,i,sx,sy,sz,0,0,0)
                   + GET(src,i,sx,sy,sz,0,0,-1)
                   + GET(src,i,sx,sy,sz,0,0,+1);
    float im_dxdy = (GET(src,i,sx,sy,sz,-1,-1,0) + GET(src,i,sx,sy,sz,1,1,0)
                   - GET(src,i,sx,sy,sz,-1,1,0) - GET(src,i,sx,sy,sz,1,-1,0)) * 0.25f;
    float im_dxdz = (GET(src,i,sx,sy,sz,-1,0,-1) + GET(src,i,sx,sy,sz,1,0,1)
                   - GET(src,i,sx,sy,sz,-1,0,1) - GET(src,i,sx,sy,sz,1,0,-1)) * 0.25f;
    float im_dydz = (GET(src,i,sx,sy,sz,0,-1,-1) + GET(src,i,sx,sy,sz,0,1,1)
                   - GET(src,i,sx,sy,sz,0,1,-1) - GET(src,i,sx,sy,sz,0,-1,1)) * 0.25f;

    const float A11 = im_dx2, A22 = im_dy2, A33 = im_dz2;
    const float A12 = im_dxdy, A13 = im_dxdz, A23 = im_dydz;

    float eig1, eig2, eig3;
    float p1 = A12*A12 + A13*A13 + A23*A23;
    if( p1 < 1e-7f ) {
        eig1 = A11; eig2 = A22; eig3 = A33;
    } else {
        float q = (A11+A22+A33)/3;
        float p2 = (A11-q)*(A11-q)+(A22-q)*(A22-q)+(A33-q)*(A33-q)+2*p1;
        float p = sqrtf(p2/6);
        float B11=(1/p)*(A11-q), B12=(1/p)*(A12-q), B13=(1/p)*(A13-q),
              B22=(1/p)*(A22-q), B23=(1/p)*(A23-q), B33=(1/p)*(A33-q);
        float B21=B12, B31=B13, B32=B23;
        float detB = B11*(B22*B33-B23*B32) - B12*(B21*B33-B23*B31) + B13*(B21*B32-B22*B31);
        float r = detB/2, phi;
        const float M_PI3 = 3.14159265f / 3;
        if(r <= -1.0f) phi = M_PI3;
        else if(r >= 1.0f) phi = 0;
        else phi = acosf(r) / 3;
        eig1 = q + 2*p*cosf(phi);
        eig3 = q + 2*p*cosf(phi+2*M_PI3);
        eig2 = 3*q - eig1 - eig3;
    }
    // sort abs(eig1) < abs(eig2) < abs(eig3)
    if(fabsf(eig1)>fabsf(eig2)){float t=eig2; eig2=eig1; eig1=t;}
    if(fabsf(eig2)>fabsf(eig3)){float t=eig2; eig2=eig3; eig3=t;}
    if(fabsf(eig1)>fabsf(eig2)){float t=eig2; eig2=eig1; eig1=t;}
    float vn = 0.0f;
    if(eig2<0 && eig3<0){
        float l1=fabsf(eig1), l2=fabsf(eig2), l3=fabsf(eig3);
        float A=l2/l3, B=l1/sqrtf(l2*l3), S=sqrtf(l1*l1+l2*l2+l3*l3);
        vn = (1.0f-expf(-A*A/alpha))*expf(B*B/beta)*(1-expf(-S*S/gamma));
    }
    if(vn>dst[i]) dst[i]=vn;
}

// =============== GAUSSIAN KERNEL GENERATION (HOST) =================

void makeGaussianKernel1D(std::vector<float>& kernel, int ksize, float sigma) {
    int half = ksize / 2;
    kernel.resize(ksize);
    float sum = 0.0f;
    for(int i=0;i<ksize;++i) {
        int x = i-half;
        float val = expf(-0.5f*x*x/(sigma*sigma));
        kernel[i]=val; sum+=val;
    }
    for(int i=0;i<ksize;++i) kernel[i] /= sum;
}
void makeGaussianKernel3D(std::vector<float>& kernel, int kx, int ky, int kz, float sigmax, float sigmay, float sigmaz) {
    std::vector<float> kxv, kyv, kzv;
    makeGaussianKernel1D(kxv, kx, sigmax);
    makeGaussianKernel1D(kyv, ky, sigmay);
    makeGaussianKernel1D(kzv, kz, sigmaz);
    kernel.resize(kx*ky*kz);
    for(int z=0;z<kz;++z)
        for(int y=0;y<ky;++y)
            for(int x=0;x<kx;++x)
                kernel[x + y*kx + z*kx*ky] = kxv[x] * kyv[y] * kzv[z];
}

// ======================= HOST: Multi-scale Vesselness ===========================
void vesselness_multiscale(
    const float* h_src, float* h_dst,
    int sx, int sy, int sz,
    float sigma_from, float sigma_to, float sigma_step,
    float alpha, float beta, float gamma)
{
    size_t N = sx * sy * sz;
    float* d_src = nullptr; float* d_blur = nullptr; float* d_temp = nullptr; float* d_dst = nullptr;
    float* d_kernel = nullptr;

    CHECK_CUDA(cudaMalloc(&d_src, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blur, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, N*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_dst, 0, N*sizeof(float)));

    int nTPB = 512;
    int nBlocks = int((N+nTPB-1)/nTPB);

    for(float sigma = sigma_from; sigma <= sigma_to+1e-4f; sigma += sigma_step) {
        int ksize = int(6*sigma+1); if(ksize%2==0) ++ksize;
        std::vector<float> h_kernel;
        makeGaussianKernel3D(h_kernel, ksize, 1, 1, sigma, 1.0f, 1.0f); // X
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), ksize*sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<nBlocks,nTPB>>>(d_src, d_blur, d_kernel, sx, sy, sz, ksize, 1, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        makeGaussianKernel3D(h_kernel, 1, ksize, 1, 1.0f, sigma, 1.0f); // Y
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), ksize*sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<nBlocks,nTPB>>>(d_blur, d_temp, d_kernel, sx, sy, sz, 1, ksize, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        makeGaussianKernel3D(h_kernel, 1, 1, ksize, 1.0f, 1.0f, sigma); // Z
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), ksize*sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<nBlocks,nTPB>>>(d_temp, d_blur, d_kernel, sx, sy, sz, 1, 1, ksize);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        multiply_kernel<<<nBlocks,nTPB>>>(d_blur, d_blur, int(N), sigma);
        CHECK_CUDA(cudaDeviceSynchronize());

        vesselness_kernel<<<nBlocks, nTPB>>>(d_blur, d_dst, sx, sy, sz, alpha, beta, gamma);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_src); cudaFree(d_blur); cudaFree(d_temp); cudaFree(d_dst);
}

// ============================= MEX ENTRY =========================================
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if(nrhs < 1)
        mexErrMsgIdAndTxt("vesselness:input", "Usage: out = vesselness_mex(volume, sigma_from, sigma_to, sigma_step, alpha, beta, gamma)");
    if(mxGetClassID(prhs[0]) != mxSINGLE_CLASS || mxGetNumberOfDimensions(prhs[0])!=3)
        mexErrMsgIdAndTxt("vesselness:input", "Input must be 3D single array");

    const mwSize* dims = mxGetDimensions(prhs[0]);
    int sx = int(dims[0]), sy = int(dims[1]), sz = int(dims[2]);
    const float* src = static_cast<const float*>(mxGetData(prhs[0]));

    float sigma_from = (nrhs>1) ? float(mxGetScalar(prhs[1])) : 1.0f;
    float sigma_to   = (nrhs>2) ? float(mxGetScalar(prhs[2]))  : 3.0f;
    float sigma_step = (nrhs>3) ? float(mxGetScalar(prhs[3]))  : 0.5f;
    float alpha      = (nrhs>4) ? float(mxGetScalar(prhs[4]))  : 1.0e-1f;
    float beta       = (nrhs>5) ? float(mxGetScalar(prhs[5]))  : 5.0f;
    float gamma      = (nrhs>6) ? float(mxGetScalar(prhs[6]))  : 3.5e5f;

    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    float* dst = static_cast<float*>(mxGetData(plhs[0]));
    vesselness_multiscale(src, dst, sx, sy, sz, sigma_from, sigma_to, sigma_step, alpha, beta, gamma);
}
