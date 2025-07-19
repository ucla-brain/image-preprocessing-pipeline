// vesselness_mex.cu -- MATLAB MEX (CUDA) for 3D Frangi/Vesselness filter (single-precision, all-in-one)
// Compile with: mexcuda -largeArrayDims vesselness_mex.cu
#include "mex.h"
#include "gpu/mxGPUArray.h"
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

__device__ __forceinline__
float get(const float* v,int i,int sx,int sy,int sz,int ox,int oy,int oz)
{
    return v[i + ox + oy*sx + oz*sx*sy];
}

#ifndef M_PI                /* π if <cmath> didn’t supply it             */
#define M_PI 3.14159265358979323846
#endif


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

/* ------------------------------------------------------------------
 * src, dst : device pointers (single)
 * sx,sy,sz : volume dimensions
 * alpha,beta,gamma : Frangi parameters
 * bright    : true = bright tubes, false = dark tubes
 * -----------------------------------------------------------------*/
__global__ void vesselness_kernel(
        const float* __restrict__ src,
              float*              dst,
        int  sx,int  sy,int  sz,
        float alpha,float beta,float gamma,
        bool  bright)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int N   = sx*sy*sz;
    if(idx>=N) return;

    /* skip borders */
    int ix =  idx % sx;
    int iy = (idx / sx) % sy;
    int iz =  idx / (sx*sy);
    if(ix<=0||ix>=sx-1||iy<=0||iy>=sy-1||iz<=0||iz>=sz-1) return;

    /* second-order derivatives */
    float dxx = -2.0f*get(src,idx,sx,sy,sz,0,0,0)
                + get(src,idx,sx,sy,sz,-1,0,0) + get(src,idx,sx,sy,sz,1,0,0);
    float dyy = -2.0f*get(src,idx,sx,sy,sz,0,0,0)
                + get(src,idx,sx,sy,sz,0,-1,0)+ get(src,idx,sx,sy,sz,0,1,0);
    float dzz = -2.0f*get(src,idx,sx,sy,sz,0,0,0)
                + get(src,idx,sx,sy,sz,0,0,-1)+ get(src,idx,sx,sy,sz,0,0,1);
    float dxy = 0.25f*( get(src,idx,sx,sy,sz,-1,-1,0)+get(src,idx,sx,sy,sz,1,1,0)
                       -get(src,idx,sx,sy,sz,-1,1,0) -get(src,idx,sx,sy,sz,1,-1,0) );
    float dxz = 0.25f*( get(src,idx,sx,sy,sz,-1,0,-1)+get(src,idx,sx,sy,sz,1,0,1)
                       -get(src,idx,sx,sy,sz,-1,0,1) -get(src,idx,sx,sy,sz,1,0,-1) );
    float dyz = 0.25f*( get(src,idx,sx,sy,sz,0,-1,-1)+get(src,idx,sx,sy,sz,0,1,1)
                       -get(src,idx,sx,sy,sz,0,1,-1) -get(src,idx,sx,sy,sz,0,-1,1) );

    /* eigenvalues of symmetric 3×3 */
    float A11=dxx,A22=dyy,A33=dzz,A12=dxy,A13=dxz,A23=dyz;
    float eig1,eig2,eig3;
    float p1 = A12*A12 + A13*A13 + A23*A23;
    if(p1<1e-7f){ eig1=A11; eig2=A22; eig3=A33; }
    else{
        float q  = (A11+A22+A33)/3.0f;
        float p2 = (A11-q)*(A11-q)+(A22-q)*(A22-q)+(A33-q)*(A33-q)+2*p1;
        float p  = sqrtf(p2/6.0f);
        float B11=(A11-q)/p, B22=(A22-q)/p, B33=(A33-q)/p;
        float B12=A12/p,     B13=A13/p,     B23=A23/p;
        float detB = B11*(B22*B33-B23*B23) - B12*(B12*B33-B23*B13)
                   + B13*(B12*B23-B22*B13);
        float r   = detB*0.5f;
        float phi = (r<=-1.f)? (float)(M_PI/3.0)
                  : (r>= 1.f)? 0.f
                  : acosf(r)/3.0f;
        eig1 = q + 2*p*cosf(phi);
        eig3 = q + 2*p*cosf(phi + 2.f*(float)M_PI/3.f);
        eig2 = 3*q - eig1 - eig3;
    }
    /* sort by absolute value */
    if(fabsf(eig1)>fabsf(eig2)){ float t=eig1; eig1=eig2; eig2=t; }
    if(fabsf(eig2)>fabsf(eig3)){ float t=eig2; eig2=eig3; eig3=t; }
    if(fabsf(eig1)>fabsf(eig2)){ float t=eig1; eig1=eig2; eig2=t; }

    bool cond = bright ? (eig2<0.f && eig3<0.f)
                       : (eig2>0.f && eig3>0.f);
    if(!cond) return;

    float l1=fabsf(eig1), l2=fabsf(eig2), l3=fabsf(eig3);
    float A=l2/l3, B=l1/sqrtf(l2*l3), S=sqrtf(l1*l1+l2*l2+l3*l3);
    float vn=(1.f-expf(-A*A/alpha))*expf(  B*B/beta)*(1.f-expf(-S*S/gamma));
    if(vn>dst[idx]) dst[idx]=vn;
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
// -----------------------------------------------------------------------------
// d_src : device pointer to input volume  (read-only)
// d_dst : device pointer to output volume (already allocated)
// -----------------------------------------------------------------------------
void vesselness_multiscale_device(
        const float* d_src, float* d_dst,
        int sx, int sy, int sz,
        float sigma_from, float sigma_to, float sigma_step,
        float alpha, float beta, float gamma,
        bool bright)                               // <── NEW
{
    const size_t N = static_cast<size_t>(sx) * sy * sz;
    float *d_blur = nullptr, *d_temp = nullptr, *d_kernel = nullptr;

    CHECK_CUDA(cudaMalloc(&d_blur, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dst, 0, N * sizeof(float)));

    const int nTPB = 512;
    const int blocks = static_cast<int>((N + nTPB - 1) / nTPB);

    for (float sigma = sigma_from; sigma <= sigma_to + 1e-4f; sigma += sigma_step) {
        int ksize = static_cast<int>(6 * sigma + 1); if (!(ksize & 1)) ++ksize;
        std::vector<float> hK;

        // X
        makeGaussianKernel3D(hK, ksize, 1, 1, sigma, 1.f, 1.f);
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, hK.data(), ksize * sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<blocks, nTPB>>>(d_src, d_blur, d_kernel, sx, sy, sz, ksize, 1, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        // Y
        makeGaussianKernel3D(hK, 1, ksize, 1, 1.f, sigma, 1.f);
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, hK.data(), ksize * sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<blocks, nTPB>>>(d_blur, d_temp, d_kernel, sx, sy, sz, 1, ksize, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        // Z
        makeGaussianKernel3D(hK, 1, 1, ksize, 1.f, 1.f, sigma);
        CHECK_CUDA(cudaMalloc(&d_kernel, ksize * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, hK.data(), ksize * sizeof(float), cudaMemcpyHostToDevice));
        cov3_kernel<<<blocks, nTPB>>>(d_temp, d_blur, d_kernel, sx, sy, sz, 1, 1, ksize);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_kernel);

        multiply_kernel<<<blocks, nTPB>>>(d_blur, d_blur, static_cast<int>(N), sigma);
        CHECK_CUDA(cudaDeviceSynchronize());

        vesselness_kernel<<<blocks, nTPB>>>(d_blur, d_dst, sx, sy, sz, alpha, beta, gamma, bright);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaFree(d_blur);
    cudaFree(d_temp);
}

// ========== MEX entry: gpuArray(single) only ==========
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();
    if(nrhs<1)
        mexErrMsgIdAndTxt("fibermetric_gpu:input",
           "Usage: out = fibermetric_gpu(gpuArray(single), [sigma_from, sigma_to, sigma_step, alpha, beta, gamma, polarity])");

    /* ---- input must be gpuArray(single) 3-D ---- */
    if(!mxIsGPUArray(prhs[0]))
        mexErrMsgIdAndTxt("fibermetric_gpu:input","Input must be gpuArray(single).");

    mxGPUArray const* gIn = mxGPUCreateFromMxArray(prhs[0]);
    if(mxGPUGetClassID(gIn)!=mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(gIn)!=3)
        mexErrMsgIdAndTxt("fibermetric_gpu:input","Input must be 3-D gpuArray(single).");

    const mwSize* d = mxGPUGetDimensions(gIn);
    int sx=d[0], sy=d[1], sz=d[2];

    /* ---- optional parameters ---- */
    float sigma_from = (nrhs>1)? (float)mxGetScalar(prhs[1]) : 1.f;
    float sigma_to   = (nrhs>2)? (float)mxGetScalar(prhs[2]) : 4.f;
    float sigma_step = (nrhs>3)? (float)mxGetScalar(prhs[3]) : 1.f;
    float alpha      = (nrhs>4)? (float)mxGetScalar(prhs[4]) : 0.1f;
    float beta       = (nrhs>5)? (float)mxGetScalar(prhs[5]) : 5.f;
    float gamma      = (nrhs>6)? (float)mxGetScalar(prhs[6]) : 3.5e5f;
    bool  bright     = true;
    if(nrhs>7){
        char* s = mxArrayToString(prhs[7]);
#ifdef _WIN32
        if(!_stricmp(s,"dark"))   bright=false;
        else if(_stricmp(s,"bright"))
#else
        if(strcasecmp(s,"dark")==0) bright=false;
        else if(strcasecmp(s,"bright")!=0)
#endif
            mexErrMsgIdAndTxt("fibermetric_gpu:polarity","polarity must be 'bright' or 'dark'");
        mxFree(s);
    }

    /* ---- allocate output gpuArray ---- */
    mwSize outDims[3]={d[0],d[1],d[2]};
    mxGPUArray* gOut = mxGPUCreateGPUArray(3,outDims,mxSINGLE_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE);
    const float* d_src = static_cast<const float*>(mxGPUGetDataReadOnly(gIn));
    float*       d_dst = static_cast<float*>(mxGPUGetData(gOut));

    /* ---- run multiscale (device-only) ---- */
    vesselness_multiscale_device(d_src,d_dst,sx,sy,sz,
        sigma_from,sigma_to,sigma_step,alpha,beta,gamma,bright);

    plhs[0] = mxGPUCreateMxArrayOnGPU(gOut);
    mxGPUDestroyGPUArray(gOut);
    mxGPUDestroyGPUArray(gIn);
}
