// fast_twin_tail_orderstat.cu
#include "mex.h"
#include <queue>
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <functional>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif


// --- Robust min_max for any numeric type ---
template<typename T>
inline std::pair<T, T> min_max(const T& a, const T& b) {
    if (a < b)
        return {a, b};
    else
        return {b, a};
}

#include <vector>
#include <algorithm>
#include <cmath>

// --- Main percentile logic ---
template<typename InputType>
void computeTwinTailPercentilesWithFixedHeapSizes(
    const InputType* inputArray,
    mwSize numElements,
    double percentileValue1,
    double percentileValue2,
    double& percentileLowResult,
    double& percentileHighResult)
{
    auto [lowestPercentile, highestPercentile] = min_max(percentileValue1, percentileValue2);

    /* ---------- ranks calculation ---------- */
    double  lowRank = 1 +  lowestPercentile / 100.0 * (numElements - 1);
    double highRank = 1 + highestPercentile / 100.0 * (numElements - 1);

    mwSize kLow  = static_cast<mwSize>( std::floor(lowRank ) );
    mwSize kHigh = static_cast<mwSize>( std::floor(highRank) );

    /* keep both x_k and x_{k+1} */
    mwSize  numLowTailHeapElements = std::max<mwSize>(2,                                               kLow + 1  ); // keep k  & k+1
    mwSize numHighTailHeapElements = std::max<mwSize>(2, std::min<mwSize>( numElements, numElements - kHigh + 1 ));

    // Max-heap for low tail (find kth smallest)
    std::priority_queue<InputType> lowTailMaxHeap;
    // Min-heap for high tail (find kth largest)
    std::priority_queue<InputType, std::vector<InputType>, std::greater<InputType>> highTailMinHeap;

    for (mwSize i = 0; i < numElements; ++i) {
        InputType v = inputArray[i];
        // Low tail (max-heap)
        if (lowTailMaxHeap.size() < numLowTailHeapElements)
            lowTailMaxHeap.push(v);
        else if (v < lowTailMaxHeap.top()) {
            lowTailMaxHeap.pop();
            lowTailMaxHeap.push(v);
        }
        // High tail (min-heap)
        if (highTailMinHeap.size() < numHighTailHeapElements)
            highTailMinHeap.push(v);
        else if (v > highTailMinHeap.top()) {
            highTailMinHeap.pop();
            highTailMinHeap.push(v);
        }
    }

    auto interpolateFromMaxHeap = [&](auto& h, double rank)->double
    {
        double v1 = static_cast<double>(h.top()); h.pop(); // (k+1)-st
        double v0 = static_cast<double>(h.top());          // k-th
        double frac = rank - std::floor(rank);
        return (1.0 - frac) * v0 + frac * v1;
    };

    auto interpolateFromMinHeap = [&](auto& h, double rank)->double
    {
        double v0 = static_cast<double>(h.top()); h.pop(); // k-th largest
        double v1 = static_cast<double>(h.top());          // (k+1)-st largest
        double frac = rank - std::floor(rank);
        return (1.0 - frac) * v0 + frac * v1;
    };

    // ----- interpolate -----
    percentileLowResult  = interpolateFromMaxHeap( lowTailMaxHeap,  lowRank);
    percentileHighResult = interpolateFromMinHeap(highTailMinHeap, highRank);
}

// --- Data type dispatcher ---
template<typename InputType>
void selectInputTypeAndComputePercentiles(
    const mxArray* inputArray, double percentile1, double percentile2, double* output)
{
    mwSize numElements = mxGetNumberOfElements(inputArray);
    const InputType* inputPtr = reinterpret_cast<const InputType*>(mxGetData(inputArray));
    computeTwinTailPercentilesWithFixedHeapSizes(inputPtr, numElements, percentile1, percentile2, output[0], output[1]);
}

// --- Entry point ---
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Usage: out = twin_percentile_mex(data, [p1, p2])");
    const mxArray* inputArray = prhs[0];
    double* percentileArray = mxGetDoubles(prhs[1]);
    int numPercentiles = static_cast<int>(mxGetNumberOfElements(prhs[1]));
    if (numPercentiles != 2)
        mexErrMsgTxt("Second argument must be a vector [p1, p2]");
    double percentile1 = percentileArray[0];
    double percentile2 = percentileArray[1];
    if (percentile1 < 0.0 || percentile1 > 100.0 || percentile2 < 0.0 || percentile2 > 100.0)
        mexErrMsgTxt("Percentiles must be between 0 and 100");

    plhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    double* output = mxGetDoubles(plhs[0]);

    switch (mxGetClassID(inputArray)) {
        case mxDOUBLE_CLASS: selectInputTypeAndComputePercentiles<double>(inputArray, percentile1, percentile2, output); break;
        case mxSINGLE_CLASS: selectInputTypeAndComputePercentiles<float>(inputArray, percentile1, percentile2, output); break;
        case mxUINT8_CLASS:  selectInputTypeAndComputePercentiles<uint8_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT8_CLASS:   selectInputTypeAndComputePercentiles<int8_t>(inputArray, percentile1, percentile2, output); break;
        case mxUINT16_CLASS: selectInputTypeAndComputePercentiles<uint16_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT16_CLASS:  selectInputTypeAndComputePercentiles<int16_t>(inputArray, percentile1, percentile2, output); break;
        case mxUINT32_CLASS: selectInputTypeAndComputePercentiles<uint32_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT32_CLASS:  selectInputTypeAndComputePercentiles<int32_t>(inputArray, percentile1, percentile2, output); break;
        case mxUINT64_CLASS: selectInputTypeAndComputePercentiles<uint64_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT64_CLASS:  selectInputTypeAndComputePercentiles<int64_t>(inputArray, percentile1, percentile2, output); break;
        default:
            mexErrMsgTxt("Unsupported class: Only real numeric arrays supported.");
    }
}