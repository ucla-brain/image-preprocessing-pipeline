// ===============================
// wavelet_filters.cuh (gpuArray-compatible)
// ===============================
#ifndef WAVELET_FILTERS_CUH
#define WAVELET_FILTERS_CUH

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>

// Define wavelet filter coefficients for Daubechies wavelets
// Source: MATLAB Wavelet Toolbox / WaveLab equivalent

inline bool get_analysis_filters(const std::string& name, std::vector<float>& Lo_D, std::vector<float>& Hi_D) {
    static std::map<std::string, std::vector<float>> wavelets = {
        {"db1", { 0.70710678f,  0.70710678f }},
        {"db4", { -0.0105974f,  0.0328830f,  0.0308414f, -0.1870350f,
                  -0.0279837f,  0.6308808f,  0.7148466f,  0.2303778f }},
        {"db9", {  0.0007701598f,  0.0000956327f, -0.0086412993f, -0.0014653826f,
                   0.0459272392f,  0.0116098939f, -0.1594942789f, -0.0708805358f,
                   0.4716909131f,  0.7695100370f,  0.3838267611f, -0.0355367405f,
                  -0.0319900568f,  0.0499949721f,  0.0057649120f, -0.0203549398f,
                  -0.0008043589f,  0.0045931736f }}
    };

    auto it = wavelets.find(name);
    if (it == wavelets.end()) return false;

    Lo_D = it->second;

    Hi_D.resize(Lo_D.size());
    for (size_t i = 0; i < Lo_D.size(); ++i) {
        Hi_D[i] = ((i & 1) ? 1 : -1) * Lo_D[Lo_D.size() - 1 - i];
    }
    return true;
}

inline bool get_synthesis_filters(const std::string& name, std::vector<float>& Lo_R, std::vector<float>& Hi_R) {
    std::vector<float> Lo_D, Hi_D;
    if (!get_analysis_filters(name, Lo_D, Hi_D)) return false;

    Lo_R.resize(Lo_D.size());
    Hi_R.resize(Hi_D.size());

    for (size_t i = 0; i < Lo_D.size(); ++i) {
        Lo_R[i] = Lo_D[Lo_D.size() - 1 - i];
        Hi_R[i] = Hi_D[Hi_D.size() - 1 - i];
    }
    return true;
}

#endif // WAVELET_FILTERS_CUH