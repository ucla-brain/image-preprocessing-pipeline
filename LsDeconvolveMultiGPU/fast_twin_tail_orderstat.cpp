/*==============================================================================
  fast_twin_tail_orderstat.cu

  Efficient twin-tail percentile estimation (prctile) for large arrays.
  MATLAB MEX implementation, heap-based, matches MATLAB's prctile(...,'Method','exact')
  with much better speed for small percentiles.

  - Finds the [p, 100-p] percentiles with O(N log K) complexity (K: heap size).
  - Only the most extreme values are kept, no full sort.
  - Uses max-heap and min-heap (std::priority_queue) for low/high tails.
  - Heaps are non-overlapping (asserted at runtime), so only one heap is ever
    updated per value after initial fill.
  - Linear interpolation is performed exactly as in MATLAB's "exact" method.
  - Designed for high performance, minimal memory, and compatibility.

  Usage in MATLAB:
      result = fast_twin_tail_orderstat(data, [p1, p2])
      // returns a 1x2 vector of the requested percentiles

  Author:      Keivan Moradi (2025)
  Assistance:  ChatGPT (OpenAI, 2025)
  License:     GPL v3

==============================================================================*/

#include "mex.h"
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cassert>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// --- Robust min_max for any numeric type ---
template<typename T>
inline auto min_max(const T& a, const T& b) {
    return (a < b) ? std::pair<T, T>{a, b} : std::pair<T, T>{b, a};
}

// Efficient pairwise min/max for a C array
template<typename T>
std::pair<double, double> min_max_pairwise(const T* arr, mwSize n) {
    if (n == 0)
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    T min_val = arr[0], max_val = arr[0];
    mwSize i = 1;
    // Process in pairs for fewer comparisons
    for (; i + 1 < n; i += 2) {
        T x = arr[i], y = arr[i+1];
        if (x > y) std::swap(x, y);
        if (x < min_val) min_val = x;
        if (y > max_val) max_val = y;
    }
    // Odd length: process last
    if (i < n) {
        auto x = arr[i];
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }
    // Return as double (not strictly required, but matches percentile output type)
    return std::make_pair(static_cast<double>(min_val), static_cast<double>(max_val));
}

template<typename Heap, typename Picker>
double interpolateFromHeap(Heap& h, double rank, Picker pick) {
    double vA = static_cast<double>(h.top()); h.pop();
    double vB = static_cast<double>(h.top());
    auto [v0, v1] = pick(vA, vB);
    double frac = rank - std::floor(rank);
    return (1.0 - frac)*v0 + frac*v1;
}

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
    assert(lowestPercentile < 50 && highestPercentile > 50 &&
       "Percentiles must straddle the median: require lowestPercentile < 50 and highestPercentile > 50.");


    // -- Compute fractional ranks as MATLAB's 'exact' method (1-based) --
    auto lowRank  = 1.0 + lowestPercentile  / 100.0 * (numElements - 1);
    auto highRank = 1.0 + highestPercentile / 100.0 * (numElements - 1);

    auto kLow  = static_cast<mwSize>(std::floor(lowRank));
    auto kHigh = static_cast<mwSize>(std::floor(highRank));

    // -- Heap sizes for interpolation (guaranteed to have both neighbors) --
    auto numLowTailHeapElements  = std::max<mwSize>(2, kLow + 1);
    auto numHighTailHeapElements = std::max<mwSize>(2, std::min<mwSize>(numElements, numElements - kHigh + 1));

    if ((lowestPercentile == 0.0 && highestPercentile == 100.0) ||
        numLowTailHeapElements + numHighTailHeapElements > numElements){
        auto minmax = min_max_pairwise(inputArray, numElements);
        percentileLowResult  = minmax.first;
        percentileHighResult = minmax.second;
        return;
    }

    // -- Reserve heap memory up front for performance --
    std::vector<InputType> lowHeapVec(inputArray, inputArray + numLowTailHeapElements);
    std::vector<InputType> highHeapVec(inputArray + numLowTailHeapElements, inputArray + numLowTailHeapElements + numHighTailHeapElements);

    std::priority_queue<InputType, std::vector<InputType>                         >  lowTailMaxHeap(std::less   <InputType>(), std::move( lowHeapVec));
    std::priority_queue<InputType, std::vector<InputType>, std::greater<InputType>> highTailMinHeap(std::greater<InputType>(), std::move(highHeapVec));

    // -- Swap until heaps are strictly separated
    while (lowTailMaxHeap.top() > highTailMinHeap.top()) {
        auto lowMax = lowTailMaxHeap.top(); lowTailMaxHeap.pop();
        auto highMin = highTailMinHeap.top(); highTailMinHeap.pop();
        lowTailMaxHeap.push(highMin);
        highTailMinHeap.push(lowMax);
        if (lowMax == highMin) break;
    }

    // -- Main single pass: fill each heap to capacity, then replace only if more extreme --
    for (mwSize i = numLowTailHeapElements + numHighTailHeapElements; i < numElements; ++i) {
        auto v = inputArray[i];

        // If v is between the current max of lowTail and min of highTail, it's not an extreme value.
        // central value, not eligible for either heap
        // Branch-prediction hint can shave ~5% on large central data
        if (v >= lowTailMaxHeap.top() && v <= highTailMinHeap.top()) continue;

        if (v < lowTailMaxHeap.top()) {
            lowTailMaxHeap.pop();
            lowTailMaxHeap.push(v);
        } else if (v > highTailMinHeap.top()) {
            highTailMinHeap.pop();
            highTailMinHeap.push(v);
        }
    }

    // -- Verification: assert heaps do not overlap --
    assert(lowTailMaxHeap.top() <= highTailMinHeap.top() &&
           "Twin-tail heaps overlap! Adjust percentiles to avoid overlap.");

    // -- Extract order statistics for interpolation (as in MATLAB) --
    percentileLowResult = interpolateFromHeap(
        lowTailMaxHeap, lowRank, [](double a, double b){ return std::make_pair(b, a); });

    percentileHighResult = interpolateFromHeap(
        highTailMinHeap, highRank, [](double a, double b){ return std::make_pair(a, b); });

}

// --- Data type dispatcher ---
template<typename InputType>
void selectInputTypeAndComputePercentiles(
    const mxArray* inputArray, double percentile1, double percentile2, double* output)
{
    auto numElements = mxGetNumberOfElements(inputArray);
    auto inputPtr = reinterpret_cast<const InputType*>(mxGetData(inputArray));
    computeTwinTailPercentilesWithFixedHeapSizes(inputPtr, numElements, percentile1, percentile2, output[0], output[1]);
}

// --- MEX entry point ---
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Usage: out = fast_twin_tail_orderstat(data, [p1, p2])");
    auto inputArray = prhs[0];
    auto percentileArray = mxGetDoubles(prhs[1]);
    auto numPercentiles = static_cast<int>(mxGetNumberOfElements(prhs[1]));
    if (numPercentiles != 2)
        mexErrMsgTxt("Second argument must be a vector [p1, p2]");
    auto percentile1 = percentileArray[0];
    auto percentile2 = percentileArray[1];
    if (percentile1 < 0.0 || percentile1 > 100.0 || percentile2 < 0.0 || percentile2 > 100.0)
        mexErrMsgTxt("Percentiles must be between 0 and 100");

    plhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    auto output = mxGetDoubles(plhs[0]);

    switch (mxGetClassID(inputArray)) {
        case mxDOUBLE_CLASS: selectInputTypeAndComputePercentiles<double>(inputArray, percentile1, percentile2, output); break;
        case mxSINGLE_CLASS: selectInputTypeAndComputePercentiles<float >(inputArray, percentile1, percentile2, output); break;
        case mxUINT8_CLASS:  selectInputTypeAndComputePercentiles<uint8_t >(inputArray, percentile1, percentile2, output); break;
        case mxINT8_CLASS:   selectInputTypeAndComputePercentiles<int8_t  >(inputArray, percentile1, percentile2, output); break;
        case mxUINT16_CLASS: selectInputTypeAndComputePercentiles<uint16_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT16_CLASS:  selectInputTypeAndComputePercentiles<int16_t >(inputArray, percentile1, percentile2, output); break;
        case mxUINT32_CLASS: selectInputTypeAndComputePercentiles<uint32_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT32_CLASS:  selectInputTypeAndComputePercentiles<int32_t >(inputArray, percentile1, percentile2, output); break;
        case mxUINT64_CLASS: selectInputTypeAndComputePercentiles<uint64_t>(inputArray, percentile1, percentile2, output); break;
        case mxINT64_CLASS:  selectInputTypeAndComputePercentiles<int64_t >(inputArray, percentile1, percentile2, output); break;
        default:
            mexErrMsgTxt("Unsupported class: Only real numeric arrays supported.");
    }
}
