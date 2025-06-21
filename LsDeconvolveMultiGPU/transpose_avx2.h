/* transpose_avx2.h  – 16×16 unsigned byte / word blocked transpose.
 *
 * Copyright (c) Intel Corporation
 * Licensed under Apache-2.0.
 */

#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

namespace simd {

/* ---------- 16×16 uint8 -> uint8 --------------------------------------- */
static inline void transpose16x16_u8(const uint8_t* src, size_t srcStride,
                                     uint8_t* dst, size_t dstStride)
{
    /* load 16 rows into 16 XMM registers */
    __m128i r[16];
    for (int i = 0; i < 16; ++i)
        r[i] = _mm_loadu_si128((const __m128i*)(src + i * srcStride));

    /* first level: interleave pairs of rows (8-bit to 16-bit) */
    __m128i t[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i]     = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm_unpackhi_epi8(r[2 * i], r[2 * i + 1]);
    }

    /* second level: interleave 16-bit to 32-bit */
    __m128i s[16];
    for (int i = 0; i < 8; ++i) {
        s[2 * i]     = _mm_unpacklo_epi16(t[2 * i], t[2 * i + 2]);
        s[2 * i + 1] = _mm_unpackhi_epi16(t[2 * i], t[2 * i + 2]);
    }

    /* third level: interleave 32-bit to 64-bit */
    for (int i = 0; i < 8; ++i) {
        t[2 * i]     = _mm_unpacklo_epi32(s[2 * i], s[2 * i + 4]);
        t[2 * i + 1] = _mm_unpackhi_epi32(s[2 * i], s[2 * i + 4]);
    }

    /* final level: 64-bit to 128-bit and store */
    for (int i = 0; i < 8; ++i) {
        r[2 * i]     = _mm_unpacklo_epi64(t[2 * i], t[2 * i + 8]);
        r[2 * i + 1] = _mm_unpackhi_epi64(t[2 * i], t[2 * i + 8]);
        _mm_storeu_si128((__m128i*)(dst + (2 * i) * dstStride),     r[2 * i]);
        _mm_storeu_si128((__m128i*)(dst + (2 * i + 1) * dstStride), r[2 * i + 1]);
    }
}

/* ---------- 16×16 uint16 -> uint16 ------------------------------------- */
static inline void transpose16x16_u16(const uint16_t* src, size_t srcStride,
                                      uint16_t* dst, size_t dstStride)
{
    const uint8_t*  src8 = (const uint8_t*)src;
    uint8_t*        dst8 = (uint8_t*)dst;
    transpose16x16_u8(src8, srcStride * 2, dst8, dstStride * 2);
}

} // namespace simd
