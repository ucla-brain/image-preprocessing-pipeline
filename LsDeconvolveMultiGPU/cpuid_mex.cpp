// cpuid_mex.cpp
#include "mex.h"
#if defined(_MSC_VER)
  #include <intrin.h>
  static void cpuid(int regs[4], int leaf) { __cpuid(regs, leaf); }
  static unsigned long long xgetbv(unsigned int idx) { return _xgetbv(idx); }
#else
  #include <cpuid.h>
  static void cpuid(int regs[4], int leaf) {
      __cpuid(leaf, regs[0], regs[1], regs[2], regs[3]);
  }
  // On non-MSVC you'd need inline asm or __builtin_ia32_xgetbv(0)
#endif

// check that OS has enabled XMM/YMM/ZMM state via XCR0[2:1]
static bool osSupportsAVX() {
    unsigned long long xcr0 = xgetbv(0);
    return (xcr0 & 0x6) == 0x6;
}

// CPUID leaf 7, EBX bit 5 → AVX2
static bool hasAVX2() {
    int regs[4];
    cpuid(regs, 0);
    if (regs[0] < 7) return false;
    cpuid(regs, 7);
    return (regs[1] & (1<<5)) && osSupportsAVX();  // AVX2 flag :contentReference[oaicite:0]{index=0}
}

// CPUID leaf 7, EBX bit 16 → AVX-512F
static bool hasAVX512F() {
    int regs[4];
    cpuid(regs, 0);
    if (regs[0] < 7) return false;
    cpuid(regs, 7);
    return (regs[1] & (1<<16)) && osSupportsAVX(); // AVX-512F flag :contentReference[oaicite:1]{index=1}
}

void mexFunction(int nlhs, mxArray *plhs[], int, const mxArray *[]) {
    bool a2 = hasAVX2();
    bool a5 = hasAVX512F();
    plhs[0] = mxCreateStructMatrix(1,1,0,NULL);
    mxAddField(plhs[0],"AVX2");
    mxAddField(plhs[0],"AVX512");
    mxSetField(plhs[0],0,"AVX2",  mxCreateLogicalScalar(a2));
    mxSetField(plhs[0],0,"AVX512",mxCreateLogicalScalar(a5));
}
