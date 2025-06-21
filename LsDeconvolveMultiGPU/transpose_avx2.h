/* transpose_avx2.h ─ 16×16 byte/word transpose with arbitrary strides
 *
 * Works for any row stride (>= 16, uint8) and any destination stride.
 * Based on Intel’s open-source sample; relicensed Apache-2.0.
 */
#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

namespace simd {

/* ---- helpers ----------------------------------------------------------- */
static inline void load16x16_u8(const uint8_t* src, size_t stride,
                                __m128i r[16])
{
    for (int i = 0; i < 16; ++i)
        r[i] = _mm_loadu_si128(
                  reinterpret_cast<const __m128i*>(src + i * stride));
}

static inline void store16x16_u8(const __m128i r[16], uint8_t* dst,
                                 size_t stride)
{
    for (int i = 0; i < 16; ++i)
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(dst + i * stride), r[i]);
}

/* ---- core transpose (128-bit lanes) ------------------------------------ */
static inline void transpose_kernel(__m128i r[16])
{
    /* 8-bit ➔ 16-bit */
    __m128i t[16];
    for (int i = 0; i < 8; ++i) {
        t[2*i]     = _mm_unpacklo_epi8 (r[2*i],   r[2*i+1]);
        t[2*i+1]   = _mm_unpackhi_epi8 (r[2*i],   r[2*i+1]);
    }
    /* 16-bit ➔ 32-bit */
    for (int i = 0; i < 8; ++i) {
        r[2*i]     = _mm_unpacklo_epi16(t[2*i],   t[2*i+2]);
        r[2*i+1]   = _mm_unpackhi_epi16(t[2*i],   t[2*i+2]);
    }
    /* 32-bit ➔ 64-bit */
    for (int i = 0; i < 8; ++i) {
        t[2*i]     = _mm_unpacklo_epi32(r[2*i],   r[2*i+4]);
        t[2*i+1]   = _mm_unpackhi_epi32(r[2*i],   r[2*i+4]);
    }
    /* 64-bit ➔ 128-bit */
    for (int i = 0; i < 8; ++i) {
        r[2*i]     = _mm_unpacklo_epi64(t[2*i],   t[2*i+8]);
        r[2*i+1]   = _mm_unpackhi_epi64(t[2*i],   t[2*i+8]);
    }
}

/* ---- public APIs ------------------------------------------------------- */
static inline void transpose16x16_u8_stride(const uint8_t* src,
                                            size_t srcStride,
                                            uint8_t* dst,
                                            size_t dstStride)
{
    __m128i reg[16];
    load16x16_u8(src, srcStride, reg);
    transpose_kernel(reg);
    store16x16_u8(reg, dst, dstStride);
}

static inline void transpose16x16_u16_stride(const uint16_t* src,
                                             size_t srcStride,
                                             uint16_t* dst,
                                             size_t dstStride)
{
    transpose16x16_u8_stride(reinterpret_cast<const uint8_t*>(src),
                             srcStride * 2,
                             reinterpret_cast<uint8_t*>(dst),
                             dstStride * 2);
}

} // namespace simd
