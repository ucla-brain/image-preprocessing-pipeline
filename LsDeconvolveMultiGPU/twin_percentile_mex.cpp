// twin_percentile_mex.cpp
// O(N) dual-tail percentile finder for all real numeric types
#include "mex.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <cstdint>

template<typename T>
void twin_percentiles_oned_pass(const T* data, std::size_t N, double p, double& out_lo, double& out_hi) {
    // Convert to double for accurate selection (preserves MATLAB semantics)
    std::vector<double> buf(N);
    for (std::size_t i = 0; i < N; ++i) buf[i] = static_cast<double>(data[i]);

    // Compute order statistics indices (0-based)
    double k_lo_f = (p/100.0)*(N-1);
    double k_hi_f = ((100.0-p)/100.0)*(N-1);
    std::size_t k_lo = static_cast<std::size_t>(std::floor(k_lo_f+1e-10));
    std::size_t k_hi = static_cast<std::size_t>(std::floor(k_hi_f+1e-10));

    // O(N) nth_element for both tails in a single pass: partition for min, then for max
    // Option 1: nth_element twice (still O(N) in practice, but two passes)
    // Option 2: nth_element once for lower, once for higher, but in-place and O(N)
    // But let's do both in one copy for clarity and avoid multi-threaded issues.

    // Find k_lo-th smallest (lower tail)
    std::nth_element(buf.begin(), buf.begin() + k_lo, buf.end());
    double lo = buf[k_lo];
    // Find k_hi-th smallest (upper tail)
    // Instead of sorting again, we must use a separate nth_element call,
    // because nth_element mutates buf. So we copy.
    std::vector<double> buf2 = buf;
    std::nth_element(buf2.begin(), buf2.begin() + k_hi, buf2.end());
    double hi = buf2[k_hi];

    out_lo = lo;
    out_hi = hi;
}

template<typename T>
void do_percentiles(const mxArray* A, double p, double* out) {
    const T* data = reinterpret_cast<const T*>(mxGetData(A));
    std::size_t N = mxGetNumberOfElements(A);
    twin_percentiles_oned_pass<T>(data, N, p, out[0], out[1]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("twinpct:nrhs",
            "Usage: twin_percentile_mex(data, p)   with 0 < p < 50");

    const mxArray* A = prhs[0];
    double p = mxGetScalar(prhs[1]);
    if (p <= 0.0 || p >= 50.0)
        mexErrMsgIdAndTxt("twinpct:range","p must be in (0,50).");

    std::size_t N = mxGetNumberOfElements(A);
    if (N == 0)
        mexErrMsgIdAndTxt("twinpct:empty","Input array is empty.");

    plhs[0] = mxCreateDoubleMatrix(1,2,mxREAL);
    double* out = mxGetDoubles(plhs[0]);

    switch(mxGetClassID(A)) {
    case mxDOUBLE_CLASS:
        do_percentiles<double>(A, p, out); break;
    case mxSINGLE_CLASS:
        do_percentiles<float>(A, p, out); break;
    case mxUINT8_CLASS:
        do_percentiles<uint8_t>(A, p, out); break;
    case mxINT8_CLASS:
        do_percentiles<int8_t>(A, p, out); break;
    case mxUINT16_CLASS:
        do_percentiles<uint16_t>(A, p, out); break;
    case mxINT16_CLASS:
        do_percentiles<int16_t>(A, p, out); break;
    case mxUINT32_CLASS:
        do_percentiles<uint32_t>(A, p, out); break;
    case mxINT32_CLASS:
        do_percentiles<int32_t>(A, p, out); break;
    case mxUINT64_CLASS:
        do_percentiles<uint64_t>(A, p, out); break;
    case mxINT64_CLASS:
        do_percentiles<int64_t>(A, p, out); break;
    default:
        mexErrMsgIdAndTxt("twinpct:type",
            "Unsupported input type: only real numeric arrays supported.");
    }
}
