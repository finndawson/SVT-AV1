/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "EbDefinitions.h"
#include <immintrin.h>
#include "convolve.h"
#include "aom_dsp_rtcd.h"
#include "convolve_avx2.h"
#include "EbInterPrediction.h"
#include "EbMemory_AVX2.h"
#include "synonyms.h"

void eb_av1_convolve_y_sr_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst, int32_t dst_stride, int32_t w,
    int32_t h, InterpFilterParams *filter_params_x,
    InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn,
    const int32_t subpel_y_qn,
    ConvolveParams *conv_params) {
    int32_t i, j;
    __m128i coeffs_128[4];
    __m256i coeffs_256[4];

    (void)filter_params_x;
    (void)subpel_x_qn;
    (void)conv_params;

    if (is_convolve_2tap(filter_params_y->filter_ptr)) {
        // vert_filt as 2 tap
        const uint8_t *src_ptr = src;

        if (subpel_y_qn != 8) {
            if (w <= 8) {
                prepare_half_coeffs_2tap_ssse3(
                    filter_params_y, subpel_y_qn, coeffs_128);

                if (w == 2) {
                    __m128i s_16[2], s_128[2];

                    s_16[0] = _mm_cvtsi32_si128(*(int16_t *)src_ptr);

                    i = h;
                    do {
                        s_16[1] = _mm_cvtsi32_si128(
                            *(int16_t *)(src_ptr + src_stride));
                        s_128[0] = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                        s_16[0] = _mm_cvtsi32_si128(
                            *(int16_t *)(src_ptr + 2 * src_stride));
                        s_128[1] = _mm_unpacklo_epi16(s_16[1], s_16[0]);
                        const __m128i ss =
                            _mm_unpacklo_epi8(s_128[0], s_128[1]);
                        const __m128i res =
                            convolve_2tap_ssse3(&ss, coeffs_128);
                        const __m128i r = convolve_y_round_sse2(res);
                        convolve_store_2x2_sse2(r, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 4) {
                    __m128i s_32[2], s_128[2];

                    s_32[0] = _mm_cvtsi32_si128(*(int32_t *)src_ptr);

                    i = h;
                    do {
                        s_32[1] = _mm_cvtsi32_si128(
                            *(int32_t *)(src_ptr + src_stride));
                        s_128[0] = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                        s_32[0] = _mm_cvtsi32_si128(
                            *(int32_t *)(src_ptr + 2 * src_stride));
                        s_128[1] = _mm_unpacklo_epi32(s_32[1], s_32[0]);
                        const __m128i ss =
                            _mm_unpacklo_epi8(s_128[0], s_128[1]);
                        const __m128i res =
                            convolve_2tap_ssse3(&ss, coeffs_128);
                        const __m128i r = convolve_y_round_sse2(res);
                        convolve_store_4x2_sse2(r, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else {
                    __m128i s_64[2], s_128[2];

                    assert(w == 8);

                    s_64[0] = _mm_loadl_epi64((__m128i *)src_ptr);

                    i = h;
                    do {
                        // Note: Faster than binding to AVX2 registers.
                        s_64[1] =
                            _mm_loadl_epi64((__m128i *)(src_ptr + src_stride));
                        s_128[0] = _mm_unpacklo_epi64(s_64[0], s_64[1]);
                        s_64[0] = _mm_loadl_epi64(
                            (__m128i *)(src_ptr + 2 * src_stride));
                        s_128[1] = _mm_unpacklo_epi64(s_64[1], s_64[0]);
                        const __m128i ss0 =
                            _mm_unpacklo_epi8(s_128[0], s_128[1]);
                        const __m128i ss1 =
                            _mm_unpackhi_epi8(s_128[0], s_128[1]);
                        const __m128i res0 =
                            convolve_2tap_ssse3(&ss0, coeffs_128);
                        const __m128i res1 =
                            convolve_2tap_ssse3(&ss1, coeffs_128);
                        const __m128i r0 = convolve_y_round_sse2(res0);
                        const __m128i r1 = convolve_y_round_sse2(res1);
                        const __m128i d = _mm_packus_epi16(r0, r1);
                        _mm_storel_epi64((__m128i *)dst, d);
                        _mm_storeh_epi64((__m128i *)(dst + dst_stride), d);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
            }
            else {
                prepare_half_coeffs_2tap_avx2(
                    filter_params_y, subpel_y_qn, coeffs_256);

                if (w == 16) {
                    __m128i s_128[2];
                    __m256i s_256[2];

                    s_128[0] = _mm_loadu_si128((__m128i *)src_ptr);

                    i = h;
                    do {
                        s_128[1] =
                            _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                        s_256[0] = _mm256_setr_m128i(s_128[0], s_128[1]);
                        s_128[0] = _mm_loadu_si128(
                            (__m128i *)(src_ptr + 2 * src_stride));
                        s_256[1] = _mm256_setr_m128i(s_128[1], s_128[0]);
                        const __m256i ss0 =
                            _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                        const __m256i ss1 =
                            _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                        const __m256i res0 =
                            convolve_2tap_avx2(&ss0, coeffs_256);
                        const __m256i res1 =
                            convolve_2tap_avx2(&ss1, coeffs_256);
                        const __m256i r0 = convolve_y_round_avx2(res0);
                        const __m256i r1 = convolve_y_round_avx2(res1);
                        convolve_store_16x2_avx2(r0, r1, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 32) {
                    __m256i s_256[2];

                    s_256[0] = _mm256_loadu_si256((__m256i *)src_ptr);

                    i = h;
                    do {
                        s_256[1] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride));
                        convolve_y_32_2tap_avx2(
                            s_256[0], s_256[1], coeffs_256, dst);
                        s_256[0] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride));
                        convolve_y_32_2tap_avx2(
                            s_256[1], s_256[0], coeffs_256, dst + dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 64) {
                    __m256i s_256[2][2];

                    s_256[0][0] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                    s_256[0][1] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));

                    i = h;
                    do {
                        s_256[1][0] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride));
                        s_256[1][1] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride + 32));
                        convolve_y_32_2tap_avx2(
                            s_256[0][0], s_256[1][0], coeffs_256, dst);
                        convolve_y_32_2tap_avx2(
                            s_256[0][1], s_256[1][1], coeffs_256, dst + 32);

                        s_256[0][0] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride));
                        s_256[0][1] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride + 32));
                        convolve_y_32_2tap_avx2(s_256[1][0],
                            s_256[0][0],
                            coeffs_256,
                            dst + dst_stride);
                        convolve_y_32_2tap_avx2(s_256[1][1],
                            s_256[0][1],
                            coeffs_256,
                            dst + dst_stride + 32);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else {
                    __m256i s_256[2][4];

                    assert(w == 128);

                    s_256[0][0] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                    s_256[0][1] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));
                    s_256[0][2] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 2 * 32));
                    s_256[0][3] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + 3 * 32));

                    i = h;
                    do {
                        s_256[1][0] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride));
                        s_256[1][1] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride + 1 * 32));
                        s_256[1][2] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride + 2 * 32));
                        s_256[1][3] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + src_stride + 3 * 32));
                        convolve_y_32_2tap_avx2(
                            s_256[0][0], s_256[1][0], coeffs_256, dst);
                        convolve_y_32_2tap_avx2(
                            s_256[0][1], s_256[1][1], coeffs_256, dst + 1 * 32);
                        convolve_y_32_2tap_avx2(
                            s_256[0][2], s_256[1][2], coeffs_256, dst + 2 * 32);
                        convolve_y_32_2tap_avx2(
                            s_256[0][3], s_256[1][3], coeffs_256, dst + 3 * 32);

                        s_256[0][0] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride));
                        s_256[0][1] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride + 1 * 32));
                        s_256[0][2] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride + 2 * 32));
                        s_256[0][3] = _mm256_loadu_si256(
                            (__m256i *)(src_ptr + 2 * src_stride + 3 * 32));
                        convolve_y_32_2tap_avx2(s_256[1][0],
                            s_256[0][0],
                            coeffs_256,
                            dst + dst_stride);
                        convolve_y_32_2tap_avx2(s_256[1][1],
                            s_256[0][1],
                            coeffs_256,
                            dst + dst_stride + 1 * 32);
                        convolve_y_32_2tap_avx2(s_256[1][2],
                            s_256[0][2],
                            coeffs_256,
                            dst + dst_stride + 2 * 32);
                        convolve_y_32_2tap_avx2(s_256[1][3],
                            s_256[0][3],
                            coeffs_256,
                            dst + dst_stride + 3 * 32);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
            }
        }
        else {
            // average to get half pel
            if (w <= 8) {
                if (w == 2) {
                    __m128i s_16[2];

                    s_16[0] = _mm_cvtsi32_si128(*(int16_t *)src_ptr);

                    i = h;
                    do {
                        s_16[1] = _mm_cvtsi32_si128(
                            *(int16_t *)(src_ptr + src_stride));
                        const __m128i d0 = _mm_avg_epu8(s_16[0], s_16[1]);
                        *(int16_t *)dst = (int16_t)_mm_cvtsi128_si32(d0);
                        s_16[0] = _mm_cvtsi32_si128(
                            *(int16_t *)(src_ptr + 2 * src_stride));
                        const __m128i d1 = _mm_avg_epu8(s_16[1], s_16[0]);
                        *(int16_t *)(dst + dst_stride) =
                            (int16_t)_mm_cvtsi128_si32(d1);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 4) {
                    __m128i s_32[2];

                    s_32[0] = _mm_cvtsi32_si128(*(int32_t *)src_ptr);

                    i = h;
                    do {
                        s_32[1] = _mm_cvtsi32_si128(
                            *(int32_t *)(src_ptr + src_stride));
                        const __m128i d0 = _mm_avg_epu8(s_32[0], s_32[1]);
                        xx_storel_32(dst, d0);
                        s_32[0] = _mm_cvtsi32_si128(
                            *(int32_t *)(src_ptr + 2 * src_stride));
                        const __m128i d1 = _mm_avg_epu8(s_32[1], s_32[0]);
                        xx_storel_32(dst + dst_stride, d1);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else {
                    __m128i s_64[2];

                    assert(w == 8);

                    s_64[0] = _mm_loadl_epi64((__m128i *)src_ptr);

                    i = h;
                    do {
                        // Note: Faster than binding to AVX2 registers.
                        s_64[1] =
                            _mm_loadl_epi64((__m128i *)(src_ptr + src_stride));
                        const __m128i d0 = _mm_avg_epu8(s_64[0], s_64[1]);
                        _mm_storel_epi64((__m128i *)dst, d0);
                        s_64[0] = _mm_loadl_epi64(
                            (__m128i *)(src_ptr + 2 * src_stride));
                        const __m128i d1 = _mm_avg_epu8(s_64[1], s_64[0]);
                        _mm_storel_epi64((__m128i *)(dst + dst_stride), d1);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
            }
            else if (w == 16) {
                __m128i s_128[2];

                s_128[0] = _mm_loadu_si128((__m128i *)src_ptr);

                i = h;
                do {
                    s_128[1] =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m128i d0 = _mm_avg_epu8(s_128[0], s_128[1]);
                    _mm_storeu_si128((__m128i *)dst, d0);
                    s_128[0] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                    const __m128i d1 = _mm_avg_epu8(s_128[1], s_128[0]);
                    _mm_storeu_si128((__m128i *)(dst + dst_stride), d1);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                __m256i s_256[2];

                s_256[0] = _mm256_loadu_si256((__m256i *)src_ptr);

                i = h;
                do {
                    s_256[1] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + src_stride));
                    convolve_y_32_2tap_avg(s_256[0], s_256[1], dst);
                    s_256[0] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride));
                    convolve_y_32_2tap_avg(
                        s_256[1], s_256[0], dst + dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 64) {
                __m256i s_256[2][2];

                s_256[0][0] = _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                s_256[0][1] = _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));

                i = h;
                do {
                    s_256[1][0] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + src_stride));
                    s_256[1][1] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + src_stride + 32));
                    convolve_y_32_2tap_avg(s_256[0][0], s_256[1][0], dst);
                    convolve_y_32_2tap_avg(s_256[0][1], s_256[1][1], dst + 32);

                    s_256[0][0] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride));
                    s_256[0][1] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride + 32));
                    convolve_y_32_2tap_avg(
                        s_256[1][0], s_256[0][0], dst + dst_stride);
                    convolve_y_32_2tap_avg(
                        s_256[1][1], s_256[0][1], dst + dst_stride + 32);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m256i s_256[2][4];

                assert(w == 128);

                s_256[0][0] = _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                s_256[0][1] = _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));
                s_256[0][2] = _mm256_loadu_si256((__m256i *)(src_ptr + 2 * 32));
                s_256[0][3] = _mm256_loadu_si256((__m256i *)(src_ptr + 3 * 32));

                i = h;
                do {
                    s_256[1][0] =
                        _mm256_loadu_si256((__m256i *)(src_ptr + src_stride));
                    s_256[1][1] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + src_stride + 1 * 32));
                    s_256[1][2] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + src_stride + 2 * 32));
                    s_256[1][3] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + src_stride + 3 * 32));
                    convolve_y_32_2tap_avg(s_256[0][0], s_256[1][0], dst);
                    convolve_y_32_2tap_avg(
                        s_256[0][1], s_256[1][1], dst + 1 * 32);
                    convolve_y_32_2tap_avg(
                        s_256[0][2], s_256[1][2], dst + 2 * 32);
                    convolve_y_32_2tap_avg(
                        s_256[0][3], s_256[1][3], dst + 3 * 32);

                    s_256[0][0] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride));
                    s_256[0][1] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride + 1 * 32));
                    s_256[0][2] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride + 2 * 32));
                    s_256[0][3] = _mm256_loadu_si256(
                        (__m256i *)(src_ptr + 2 * src_stride + 3 * 32));
                    convolve_y_32_2tap_avg(
                        s_256[1][0], s_256[0][0], dst + dst_stride);
                    convolve_y_32_2tap_avg(
                        s_256[1][1], s_256[0][1], dst + dst_stride + 1 * 32);
                    convolve_y_32_2tap_avg(
                        s_256[1][2], s_256[0][2], dst + dst_stride + 2 * 32);
                    convolve_y_32_2tap_avg(
                        s_256[1][3], s_256[0][3], dst + dst_stride + 3 * 32);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
        }
    }
    else if (is_convolve_4tap(filter_params_y->filter_ptr)) {
        // vert_filt as 4 tap
        const uint8_t *src_ptr = src - src_stride;

        if (w <= 4) {
            prepare_half_coeffs_4tap_ssse3(
                filter_params_y, subpel_y_qn, coeffs_128);

            if (w == 2) {
                __m128i s_16[4], ss_128[2];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_16[3] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 1 * src_stride));
                    const __m128i src23 = _mm_unpacklo_epi16(s_16[2], s_16[3]);
                    s_16[2] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 2 * src_stride));
                    const __m128i src34 = _mm_unpacklo_epi16(s_16[3], s_16[2]);
                    ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                    const __m128i res = convolve_4tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_2x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m128i s_32[4], ss_128[2];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_32[3] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 1 * src_stride));
                    const __m128i src23 = _mm_unpacklo_epi32(s_32[2], s_32[3]);
                    s_32[2] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 2 * src_stride));
                    const __m128i src34 = _mm_unpacklo_epi32(s_32[3], s_32[2]);
                    ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                    const __m128i res = convolve_4tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_4x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
        }
        else {
            prepare_half_coeffs_4tap_avx2(
                filter_params_y, subpel_y_qn, coeffs_256);

            if (w == 8) {
                __m128i s_64[4];
                __m256i ss_256[2];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_64[3] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                    const __m256i src23 = _mm256_setr_m128i(s_64[2], s_64[3]);
                    s_64[2] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));
                    const __m256i src34 = _mm256_setr_m128i(s_64[3], s_64[2]);
                    ss_256[1] = _mm256_unpacklo_epi8(src23, src34);

                    const __m256i res = convolve_4tap_avx2(ss_256, coeffs_256);
                    const __m256i r = convolve_y_round_avx2(res);
                    convolve_store_8x2_avx2(r, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m128i s_128[4];
                __m256i ss_256[4];

                assert(w == 16);

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[2] = _mm256_unpackhi_epi8(src01, src12);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_128[3] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                    const __m256i src23 = _mm256_setr_m128i(s_128[2], s_128[3]);
                    s_128[2] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                    const __m256i src34 = _mm256_setr_m128i(s_128[3], s_128[2]);
                    ss_256[1] = _mm256_unpacklo_epi8(src23, src34);
                    ss_256[3] = _mm256_unpackhi_epi8(src23, src34);

                    const __m256i res0 = convolve_4tap_avx2(ss_256, coeffs_256);
                    const __m256i res1 =
                        convolve_4tap_avx2(ss_256 + 2, coeffs_256);
                    const __m256i r0 = convolve_y_round_avx2(res0);
                    const __m256i r1 = convolve_y_round_avx2(res1);
                    convolve_store_16x2_avx2(r0, r1, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    ss_256[2] = ss_256[3];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
        }
    }
    else if (is_convolve_6tap(filter_params_y->filter_ptr)) {
        const uint8_t *src_ptr = src - 2 * src_stride;

        if (w <= 4) {
            prepare_half_coeffs_6tap_ssse3(
                filter_params_y, subpel_y_qn, coeffs_128);

            if (w == 2) {
                __m128i s_16[6], ss_128[3];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));
                s_16[3] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 3 * src_stride));
                s_16[4] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 4 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);
                const __m128i src23 = _mm_unpacklo_epi16(s_16[2], s_16[3]);
                const __m128i src34 = _mm_unpacklo_epi16(s_16[3], s_16[4]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_16[5] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 3 * src_stride));
                    const __m128i src45 = _mm_unpacklo_epi16(s_16[4], s_16[5]);
                    s_16[4] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 4 * src_stride));
                    const __m128i src56 = _mm_unpacklo_epi16(s_16[5], s_16[4]);
                    ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                    const __m128i res = convolve_6tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_2x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    ss_128[1] = ss_128[2];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m128i s_32[6], ss_128[3];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));
                s_32[3] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 3 * src_stride));
                s_32[4] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 4 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);
                const __m128i src23 = _mm_unpacklo_epi32(s_32[2], s_32[3]);
                const __m128i src34 = _mm_unpacklo_epi32(s_32[3], s_32[4]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_32[5] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 3 * src_stride));
                    const __m128i src45 = _mm_unpacklo_epi32(s_32[4], s_32[5]);
                    s_32[4] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 4 * src_stride));
                    const __m128i src56 = _mm_unpacklo_epi32(s_32[5], s_32[4]);
                    ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                    const __m128i res = convolve_6tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_4x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    ss_128[1] = ss_128[2];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
        }
        else {
            prepare_half_coeffs_6tap_avx2(
                filter_params_y, subpel_y_qn, coeffs_256);

            if (w == 8) {
                __m128i s_64[6];
                __m256i ss_256[3];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));
                s_64[3] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 3 * src_stride));
                s_64[4] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 4 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);
                const __m256i src23 = _mm256_setr_m128i(s_64[2], s_64[3]);
                const __m256i src34 = _mm256_setr_m128i(s_64[3], s_64[4]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_64[5] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 3 * src_stride));
                    const __m256i src45 = _mm256_setr_m128i(s_64[4], s_64[5]);
                    s_64[4] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 4 * src_stride));
                    const __m256i src56 = _mm256_setr_m128i(s_64[5], s_64[4]);
                    ss_256[2] = _mm256_unpacklo_epi8(src45, src56);

                    const __m256i res = convolve_6tap_avx2(ss_256, coeffs_256);
                    const __m256i r = convolve_y_round_avx2(res);
                    convolve_store_8x2_avx2(r, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    ss_256[1] = ss_256[2];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                __m128i s_128[6];
                __m256i ss_256[6];

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                s_128[3] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 3 * src_stride));
                s_128[4] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 4 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);
                const __m256i src23 = _mm256_setr_m128i(s_128[2], s_128[3]);
                const __m256i src34 = _mm256_setr_m128i(s_128[3], s_128[4]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);

                ss_256[3] = _mm256_unpackhi_epi8(src01, src12);
                ss_256[4] = _mm256_unpackhi_epi8(src23, src34);

                i = h;
                do {
                    src_ptr += 2 * src_stride;
                    s_128[5] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 3 * src_stride));
                    const __m256i src45 = _mm256_setr_m128i(s_128[4], s_128[5]);
                    s_128[4] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 4 * src_stride));
                    const __m256i src56 = _mm256_setr_m128i(s_128[5], s_128[4]);
                    ss_256[2] = _mm256_unpacklo_epi8(src45, src56);
                    ss_256[5] = _mm256_unpackhi_epi8(src45, src56);

                    const __m256i res0 = convolve_6tap_avx2(ss_256, coeffs_256);
                    const __m256i res1 =
                        convolve_6tap_avx2(ss_256 + 3, coeffs_256);
                    const __m256i r0 = convolve_y_round_avx2(res0);
                    const __m256i r1 = convolve_y_round_avx2(res1);
                    convolve_store_16x2_avx2(r0, r1, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    ss_256[1] = ss_256[2];

                    ss_256[3] = ss_256[4];
                    ss_256[4] = ss_256[5];
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m256i s_256[6], ss_256[6], tt_256[6];

                assert(!(w % 32));

                j = 0;
                do {
                    const uint8_t *s = src_ptr + j;
                    uint8_t *d = dst + j;

                    s_256[0] =
                        _mm256_loadu_si256((__m256i *)(s + 0 * src_stride));
                    s_256[1] =
                        _mm256_loadu_si256((__m256i *)(s + 1 * src_stride));
                    s_256[2] =
                        _mm256_loadu_si256((__m256i *)(s + 2 * src_stride));
                    s_256[3] =
                        _mm256_loadu_si256((__m256i *)(s + 3 * src_stride));
                    s_256[4] =
                        _mm256_loadu_si256((__m256i *)(s + 4 * src_stride));

                    ss_256[0] = _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                    ss_256[1] = _mm256_unpacklo_epi8(s_256[2], s_256[3]);
                    ss_256[3] = _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                    ss_256[4] = _mm256_unpackhi_epi8(s_256[2], s_256[3]);

                    tt_256[0] = _mm256_unpacklo_epi8(s_256[1], s_256[2]);
                    tt_256[1] = _mm256_unpacklo_epi8(s_256[3], s_256[4]);
                    tt_256[3] = _mm256_unpackhi_epi8(s_256[1], s_256[2]);
                    tt_256[4] = _mm256_unpackhi_epi8(s_256[3], s_256[4]);

                    i = h;
                    do {
                        s += 2 * src_stride;
                        s_256[5] =
                            _mm256_loadu_si256((__m256i *)(s + 3 * src_stride));
                        ss_256[2] = _mm256_unpacklo_epi8(s_256[4], s_256[5]);
                        ss_256[5] = _mm256_unpackhi_epi8(s_256[4], s_256[5]);
                        s_256[4] =
                            _mm256_loadu_si256((__m256i *)(s + 4 * src_stride));
                        tt_256[2] = _mm256_unpacklo_epi8(s_256[5], s_256[4]);
                        tt_256[5] = _mm256_unpackhi_epi8(s_256[5], s_256[4]);
                        convolve_y_32_6tap_avx2(ss_256, coeffs_256, d);
                        convolve_y_32_6tap_avx2(
                            tt_256, coeffs_256, d + dst_stride);

                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        ss_256[3] = ss_256[4];
                        ss_256[4] = ss_256[5];

                        tt_256[0] = tt_256[1];
                        tt_256[1] = tt_256[2];
                        tt_256[3] = tt_256[4];
                        tt_256[4] = tt_256[5];
                        d += 2 * dst_stride;
                        i -= 2;
                    } while (i);

                    j += 32;
                } while (j < w);
            }
        }
    }
    else {
        const uint8_t *src_ptr = src - 3 * src_stride;

        if (w <= 4) {
            prepare_half_coeffs_8tap_ssse3(
                filter_params_y, subpel_y_qn, coeffs_128);

            if (w == 2) {
                __m128i s_16[8], ss_128[4];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));
                s_16[3] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 3 * src_stride));
                s_16[4] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 4 * src_stride));
                s_16[5] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 5 * src_stride));
                s_16[6] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 6 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);
                const __m128i src23 = _mm_unpacklo_epi16(s_16[2], s_16[3]);
                const __m128i src34 = _mm_unpacklo_epi16(s_16[3], s_16[4]);
                const __m128i src45 = _mm_unpacklo_epi16(s_16[4], s_16[5]);
                const __m128i src56 = _mm_unpacklo_epi16(s_16[5], s_16[6]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);
                ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                i = h;
                do {
                    s_16[7] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 7 * src_stride));
                    const __m128i src67 = _mm_unpacklo_epi16(s_16[6], s_16[7]);
                    s_16[6] = _mm_cvtsi32_si128(
                        *(int16_t *)(src_ptr + 8 * src_stride));
                    const __m128i src78 = _mm_unpacklo_epi16(s_16[7], s_16[6]);
                    ss_128[3] = _mm_unpacklo_epi8(src67, src78);

                    const __m128i res = convolve_8tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_2x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    ss_128[1] = ss_128[2];
                    ss_128[2] = ss_128[3];
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m128i s_32[8], ss_128[4];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));
                s_32[3] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 3 * src_stride));
                s_32[4] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 4 * src_stride));
                s_32[5] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 5 * src_stride));
                s_32[6] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 6 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);
                const __m128i src23 = _mm_unpacklo_epi32(s_32[2], s_32[3]);
                const __m128i src34 = _mm_unpacklo_epi32(s_32[3], s_32[4]);
                const __m128i src45 = _mm_unpacklo_epi32(s_32[4], s_32[5]);
                const __m128i src56 = _mm_unpacklo_epi32(s_32[5], s_32[6]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);
                ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                i = h;
                do {
                    s_32[7] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 7 * src_stride));
                    const __m128i src67 = _mm_unpacklo_epi32(s_32[6], s_32[7]);
                    s_32[6] = _mm_cvtsi32_si128(
                        *(int32_t *)(src_ptr + 8 * src_stride));
                    const __m128i src78 = _mm_unpacklo_epi32(s_32[7], s_32[6]);
                    ss_128[3] = _mm_unpacklo_epi8(src67, src78);

                    const __m128i res = convolve_8tap_ssse3(ss_128, coeffs_128);
                    const __m128i r = convolve_y_round_sse2(res);
                    convolve_store_4x2_sse2(r, dst, dst_stride);

                    ss_128[0] = ss_128[1];
                    ss_128[1] = ss_128[2];
                    ss_128[2] = ss_128[3];
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
        }
        else {
            prepare_half_coeffs_8tap_avx2(
                filter_params_y, subpel_y_qn, coeffs_256);

            if (w == 8) {
                __m128i s_64[8];
                __m256i ss_256[4];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));
                s_64[3] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 3 * src_stride));
                s_64[4] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 4 * src_stride));
                s_64[5] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 5 * src_stride));
                s_64[6] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 6 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);
                const __m256i src23 = _mm256_setr_m128i(s_64[2], s_64[3]);
                const __m256i src34 = _mm256_setr_m128i(s_64[3], s_64[4]);
                const __m256i src45 = _mm256_setr_m128i(s_64[4], s_64[5]);
                const __m256i src56 = _mm256_setr_m128i(s_64[5], s_64[6]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);
                ss_256[2] = _mm256_unpacklo_epi8(src45, src56);

                i = h;
                do {
                    s_64[7] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 7 * src_stride));
                    const __m256i src67 = _mm256_setr_m128i(s_64[6], s_64[7]);
                    s_64[6] =
                        _mm_loadl_epi64((__m128i *)(src_ptr + 8 * src_stride));
                    const __m256i src78 = _mm256_setr_m128i(s_64[7], s_64[6]);
                    ss_256[3] = _mm256_unpacklo_epi8(src67, src78);

                    const __m256i res = convolve_8tap_avx2(ss_256, coeffs_256);
                    const __m256i r = convolve_y_round_avx2(res);
                    convolve_store_8x2_avx2(r, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    ss_256[1] = ss_256[2];
                    ss_256[2] = ss_256[3];
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                __m128i s_128[8];
                __m256i ss_256[8];

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                s_128[3] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 3 * src_stride));
                s_128[4] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 4 * src_stride));
                s_128[5] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 5 * src_stride));
                s_128[6] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 6 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);
                const __m256i src23 = _mm256_setr_m128i(s_128[2], s_128[3]);
                const __m256i src34 = _mm256_setr_m128i(s_128[3], s_128[4]);
                const __m256i src45 = _mm256_setr_m128i(s_128[4], s_128[5]);
                const __m256i src56 = _mm256_setr_m128i(s_128[5], s_128[6]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);
                ss_256[2] = _mm256_unpacklo_epi8(src45, src56);

                ss_256[4] = _mm256_unpackhi_epi8(src01, src12);
                ss_256[5] = _mm256_unpackhi_epi8(src23, src34);
                ss_256[6] = _mm256_unpackhi_epi8(src45, src56);

                i = h;
                do {
                    s_128[7] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 7 * src_stride));
                    const __m256i src67 = _mm256_setr_m128i(s_128[6], s_128[7]);
                    s_128[6] =
                        _mm_loadu_si128((__m128i *)(src_ptr + 8 * src_stride));
                    const __m256i src78 = _mm256_setr_m128i(s_128[7], s_128[6]);
                    ss_256[3] = _mm256_unpacklo_epi8(src67, src78);
                    ss_256[7] = _mm256_unpackhi_epi8(src67, src78);

                    const __m256i res0 = convolve_8tap_avx2(ss_256, coeffs_256);
                    const __m256i res1 =
                        convolve_8tap_avx2(ss_256 + 4, coeffs_256);
                    const __m256i r0 = convolve_y_round_avx2(res0);
                    const __m256i r1 = convolve_y_round_avx2(res1);
                    convolve_store_16x2_avx2(r0, r1, dst, dst_stride);

                    ss_256[0] = ss_256[1];
                    ss_256[1] = ss_256[2];
                    ss_256[2] = ss_256[3];

                    ss_256[4] = ss_256[5];
                    ss_256[5] = ss_256[6];
                    ss_256[6] = ss_256[7];
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else {
                __m256i s_256[8], ss_256[8], tt_256[8];

                assert(!(w % 32));

                j = 0;
                do {
                    const uint8_t *s = src_ptr + j;
                    uint8_t *d = dst + j;

                    s_256[0] =
                        _mm256_loadu_si256((__m256i *)(s + 0 * src_stride));
                    s_256[1] =
                        _mm256_loadu_si256((__m256i *)(s + 1 * src_stride));
                    s_256[2] =
                        _mm256_loadu_si256((__m256i *)(s + 2 * src_stride));
                    s_256[3] =
                        _mm256_loadu_si256((__m256i *)(s + 3 * src_stride));
                    s_256[4] =
                        _mm256_loadu_si256((__m256i *)(s + 4 * src_stride));
                    s_256[5] =
                        _mm256_loadu_si256((__m256i *)(s + 5 * src_stride));
                    s_256[6] =
                        _mm256_loadu_si256((__m256i *)(s + 6 * src_stride));

                    ss_256[0] = _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                    ss_256[1] = _mm256_unpacklo_epi8(s_256[2], s_256[3]);
                    ss_256[2] = _mm256_unpacklo_epi8(s_256[4], s_256[5]);
                    ss_256[4] = _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                    ss_256[5] = _mm256_unpackhi_epi8(s_256[2], s_256[3]);
                    ss_256[6] = _mm256_unpackhi_epi8(s_256[4], s_256[5]);

                    tt_256[0] = _mm256_unpacklo_epi8(s_256[1], s_256[2]);
                    tt_256[1] = _mm256_unpacklo_epi8(s_256[3], s_256[4]);
                    tt_256[2] = _mm256_unpacklo_epi8(s_256[5], s_256[6]);
                    tt_256[4] = _mm256_unpackhi_epi8(s_256[1], s_256[2]);
                    tt_256[5] = _mm256_unpackhi_epi8(s_256[3], s_256[4]);
                    tt_256[6] = _mm256_unpackhi_epi8(s_256[5], s_256[6]);

                    i = h;
                    do {
                        s_256[7] =
                            _mm256_loadu_si256((__m256i *)(s + 7 * src_stride));
                        ss_256[3] = _mm256_unpacklo_epi8(s_256[6], s_256[7]);
                        ss_256[7] = _mm256_unpackhi_epi8(s_256[6], s_256[7]);
                        s_256[6] =
                            _mm256_loadu_si256((__m256i *)(s + 8 * src_stride));
                        tt_256[3] = _mm256_unpacklo_epi8(s_256[7], s_256[6]);
                        tt_256[7] = _mm256_unpackhi_epi8(s_256[7], s_256[6]);
                        convolve_y_32_8tap_avx2(ss_256, coeffs_256, d);
                        convolve_y_32_8tap_avx2(
                            tt_256, coeffs_256, d + dst_stride);

                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        ss_256[2] = ss_256[3];
                        ss_256[4] = ss_256[5];
                        ss_256[5] = ss_256[6];
                        ss_256[6] = ss_256[7];

                        tt_256[0] = tt_256[1];
                        tt_256[1] = tt_256[2];
                        tt_256[2] = tt_256[3];
                        tt_256[4] = tt_256[5];
                        tt_256[5] = tt_256[6];
                        tt_256[6] = tt_256[7];
                        s += 2 * src_stride;
                        d += 2 * dst_stride;
                        i -= 2;
                    } while (i);

                    j += 32;
                } while (j < w);
            }
        }
    }
}

void eb_av1_convolve_x_sr_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst, int32_t dst_stride, int32_t w,
    int32_t h, InterpFilterParams *filter_params_x,
    InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn,
    const int32_t subpel_y_qn,
    ConvolveParams *conv_params) {
    int32_t i = h;
    __m128i coeffs_128[4];
    __m256i coeffs_256[4];

    (void)filter_params_y;
    (void)subpel_y_qn;
    (void)conv_params;

    assert(conv_params->round_0 == 3);
    assert((FILTER_BITS - conv_params->round_1) >= 0 ||
        ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

    if (is_convolve_2tap(filter_params_x->filter_ptr)) {
        // horz_filt as 2 tap
        const uint8_t *src_ptr = src;

        if (subpel_x_qn != 8) {
            if (w <= 8) {
                prepare_half_coeffs_2tap_ssse3(
                    filter_params_x, subpel_x_qn, coeffs_128);

                if (w == 2) {
                    const __m128i c = _mm_setr_epi8(
                        0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);

                    do {
                        __m128i s_128;

                        s_128 = _mm_cvtsi32_si128(*(int32_t *)src_ptr);
                        s_128 = _mm_insert_epi32(
                            s_128, *(int32_t *)(src_ptr + src_stride), 2);
                        s_128 = _mm_shuffle_epi8(s_128, c);
                        const __m128i res =
                            convolve_2tap_ssse3(&s_128, coeffs_128);
                        const __m128i r = convolve_x_round_sse2(res);
                        const __m128i d = _mm_packus_epi16(r, r);
                        *(uint16_t *)dst = (uint16_t)_mm_cvtsi128_si32(d);
                        *(uint16_t *)(dst + dst_stride) =
                            _mm_extract_epi16(d, 2);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 4) {
                    const __m128i c = _mm_setr_epi8(
                        0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);

                    do {
                        __m128i s_128;

                        s_128 = _mm_loadl_epi64((__m128i *)src_ptr);
                        s_128 = _mm_loadh_epi64(src_ptr + src_stride, s_128);
                        s_128 = _mm_shuffle_epi8(s_128, c);
                        const __m128i res =
                            convolve_2tap_ssse3(&s_128, coeffs_128);
                        const __m128i r = convolve_x_round_sse2(res);
                        convolve_store_4x2_sse2(r, dst, dst_stride);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else {
                    assert(w == 8);

                    do {
                        __m128i s_128[2], res[2];

                        const __m128i s00 = _mm_loadu_si128((__m128i *)src_ptr);
                        const __m128i s10 =
                            _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                        const __m128i s01 = _mm_srli_si128(s00, 1);
                        const __m128i s11 = _mm_srli_si128(s10, 1);
                        s_128[0] = _mm_unpacklo_epi8(s00, s01);
                        s_128[1] = _mm_unpacklo_epi8(s10, s11);

                        res[0] = convolve_2tap_ssse3(&s_128[0], coeffs_128);
                        res[1] = convolve_2tap_ssse3(&s_128[1], coeffs_128);
                        res[0] = convolve_x_round_sse2(res[0]);
                        res[1] = convolve_x_round_sse2(res[1]);
                        const __m128i d = _mm_packus_epi16(res[0], res[1]);
                        _mm_storel_epi64((__m128i *)dst, d);
                        _mm_storeh_epi64((__m128i *)(dst + dst_stride), d);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
            }
            else {
                prepare_half_coeffs_2tap_avx2(
                    filter_params_x, subpel_x_qn, coeffs_256);

                if (w == 16) {
                    do {
                        __m128i s_128[2][2];
                        __m256i s_256[2];

                        s_128[0][0] = _mm_loadu_si128((__m128i *)src_ptr);
                        s_128[0][1] = _mm_loadu_si128((__m128i *)(src_ptr + 1));
                        s_128[1][0] =
                            _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                        s_128[1][1] = _mm_loadu_si128(
                            (__m128i *)(src_ptr + src_stride + 1));
                        s_256[0] = _mm256_setr_m128i(s_128[0][0], s_128[1][0]);
                        s_256[1] = _mm256_setr_m128i(s_128[0][1], s_128[1][1]);
                        const __m256i s0 =
                            _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                        const __m256i s1 =
                            _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                        const __m256i res0 =
                            convolve_2tap_avx2(&s0, coeffs_256);
                        const __m256i res1 =
                            convolve_2tap_avx2(&s1, coeffs_256);
                        const __m256i r0 = convolve_x_round_avx2(res0);
                        const __m256i r1 = convolve_x_round_avx2(res1);
                        convolve_store_16x2_avx2(r0, r1, dst, dst_stride);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 32) {
                    do {
                        convolve_x_32_2tap_avx2(src_ptr, coeffs_256, dst);
                        src_ptr += src_stride;
                        dst += dst_stride;
                    } while (--i);
                }
                else if (w == 64) {
                    do {
                        convolve_x_32_2tap_avx2(
                            src_ptr + 0 * 32, coeffs_256, dst + 0 * 32);
                        convolve_x_32_2tap_avx2(
                            src_ptr + 1 * 32, coeffs_256, dst + 1 * 32);
                        src_ptr += src_stride;
                        dst += dst_stride;
                    } while (--i);
                }
                else {
                    assert(w == 128);

                    do {
                        convolve_x_32_2tap_avx2(
                            src_ptr + 0 * 32, coeffs_256, dst + 0 * 32);
                        convolve_x_32_2tap_avx2(
                            src_ptr + 1 * 32, coeffs_256, dst + 1 * 32);
                        convolve_x_32_2tap_avx2(
                            src_ptr + 2 * 32, coeffs_256, dst + 2 * 32);
                        convolve_x_32_2tap_avx2(
                            src_ptr + 3 * 32, coeffs_256, dst + 3 * 32);
                        src_ptr += src_stride;
                        dst += dst_stride;
                    } while (--i);
                }
            }
        }
        else {
            // average to get half pel
            if (w == 2) {
                do {
                    __m128i s_128;

                    s_128 = _mm_cvtsi32_si128(*(int32_t *)src_ptr);
                    s_128 = _mm_insert_epi32(
                        s_128, *(int32_t *)(src_ptr + src_stride), 2);
                    const __m128i s1 = _mm_srli_si128(s_128, 1);
                    const __m128i d = _mm_avg_epu8(s_128, s1);
                    *(uint16_t *)dst = (uint16_t)_mm_cvtsi128_si32(d);
                    *(uint16_t *)(dst + dst_stride) = _mm_extract_epi16(d, 4);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 4) {
                do {
                    __m128i s_128;

                    s_128 = _mm_loadl_epi64((__m128i *)src_ptr);
                    s_128 = _mm_loadh_epi64(src_ptr + src_stride, s_128);
                    const __m128i s1 = _mm_srli_si128(s_128, 1);
                    const __m128i d = _mm_avg_epu8(s_128, s1);
                    xx_storel_32(dst, d);
                    *(int32_t *)(dst + dst_stride) = _mm_extract_epi32(d, 2);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 8) {
                do {
                    const __m128i s00 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s10 =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m128i s01 = _mm_srli_si128(s00, 1);
                    const __m128i s11 = _mm_srli_si128(s10, 1);
                    const __m128i d0 = _mm_avg_epu8(s00, s01);
                    const __m128i d1 = _mm_avg_epu8(s10, s11);
                    _mm_storel_epi64((__m128i *)dst, d0);
                    _mm_storel_epi64((__m128i *)(dst + dst_stride), d1);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                do {
                    const __m128i s00 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s01 =
                        _mm_loadu_si128((__m128i *)(src_ptr + 1));
                    const __m128i s10 =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m128i s11 =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride + 1));
                    const __m128i d0 = _mm_avg_epu8(s00, s01);
                    const __m128i d1 = _mm_avg_epu8(s10, s11);
                    _mm_storeu_si128((__m128i *)dst, d0);
                    _mm_storeu_si128((__m128i *)(dst + dst_stride), d1);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                do {
                    convolve_x_32_2tap_avg(src_ptr, dst);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_x_32_2tap_avg(src_ptr + 0 * 32, dst + 0 * 32);
                    convolve_x_32_2tap_avg(src_ptr + 1 * 32, dst + 1 * 32);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_x_32_2tap_avg(src_ptr + 0 * 32, dst + 0 * 32);
                    convolve_x_32_2tap_avg(src_ptr + 1 * 32, dst + 1 * 32);
                    convolve_x_32_2tap_avg(src_ptr + 2 * 32, dst + 2 * 32);
                    convolve_x_32_2tap_avg(src_ptr + 3 * 32, dst + 3 * 32);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
    }
    else if (is_convolve_4tap(filter_params_x->filter_ptr)) {
        // horz_filt as 4 tap
        const uint8_t *src_ptr = src - 1;
        const __m128i c0 =
            _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);
        const __m128i c1 = _mm_setr_epi8(
            2, 3, 3, 4, 4, 5, 5, 6, 10, 11, 11, 12, 12, 13, 13, 14);
        __m128i t, s_128[2];

        prepare_half_coeffs_4tap_ssse3(
            filter_params_x, subpel_x_qn, coeffs_128);

        if (w == 2) {
            do {
                t = _mm_loadl_epi64((__m128i *)src_ptr);
                t = _mm_loadh_epi64(src_ptr + src_stride, t);
                s_128[0] = _mm_shuffle_epi8(t, c0);
                s_128[1] = _mm_shuffle_epi8(t, c1);
                const __m128i res = convolve_4tap_ssse3(s_128, coeffs_128);
                const __m128i r = convolve_x_round_sse2(res);
                const __m128i d = _mm_packus_epi16(r, r);
                *(uint16_t *)dst = (uint16_t)_mm_cvtsi128_si32(d);
                *(uint16_t *)(dst + dst_stride) = _mm_extract_epi16(d, 2);

                src_ptr += 2 * src_stride;
                dst += 2 * dst_stride;
                i -= 2;
            } while (i);
        }
        else {
            assert(w == 4);

            do {
                t = _mm_loadl_epi64((__m128i *)src_ptr);
                t = _mm_loadh_epi64(src_ptr + src_stride, t);
                s_128[0] = _mm_shuffle_epi8(t, c0);
                s_128[1] = _mm_shuffle_epi8(t, c1);
                const __m128i res = convolve_4tap_ssse3(s_128, coeffs_128);
                const __m128i r = convolve_x_round_sse2(res);
                convolve_store_4x2_sse2(r, dst, dst_stride);

                src_ptr += 2 * src_stride;
                dst += 2 * dst_stride;
                i -= 2;
            } while (i);
        }
    }
    else {
        __m256i filt_256[4];

        filt_256[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
        filt_256[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);
        filt_256[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);

        if (is_convolve_6tap(filter_params_x->filter_ptr)) {
            // horz_filt as 6 tap
            const uint8_t *src_ptr = src - 2;

            prepare_half_coeffs_6tap_avx2(
                filter_params_x, subpel_x_qn, coeffs_256);

            if (w == 8) {
                do {
                    const __m128i s0 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s1 =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m256i s_256 = _mm256_setr_m128i(s0, s1);
                    const __m256i res =
                        convolve_x_6tap_avx2(s_256, coeffs_256, filt_256);
                    const __m256i r = convolve_x_round_avx2(res);
                    convolve_store_8x2_avx2(r, dst, dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                do {
                    convolve_x_16x2_6tap_avx2(src_ptr,
                        src_stride,
                        coeffs_256,
                        filt_256,
                        dst,
                        dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                do {
                    convolve_x_16x2_6tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_x_16x2_6tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_x_16x2_6tap_avx2(
                        src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_x_16x2_6tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_x_16x2_6tap_avx2(
                        src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    convolve_x_16x2_6tap_avx2(
                        src_ptr + 64, 16, coeffs_256, filt_256, dst + 64, 16);
                    convolve_x_16x2_6tap_avx2(
                        src_ptr + 96, 16, coeffs_256, filt_256, dst + 96, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
        else {
            // horz_filt as 8 tap
            const uint8_t *src_ptr = src - 3;

            filt_256[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

            prepare_half_coeffs_8tap_avx2(
                filter_params_x, subpel_x_qn, coeffs_256);

            if (w == 8) {
                do {
                    const __m128i s0 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s1 =
                        _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m256i s_256 = _mm256_setr_m128i(s0, s1);
                    const __m256i res =
                        convolve_x_8tap_avx2(s_256, coeffs_256, filt_256);
                    const __m256i r = convolve_x_round_avx2(res);
                    convolve_store_8x2_avx2(r, dst, dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                do {
                    convolve_x_16x2_8tap_avx2(src_ptr,
                        src_stride,
                        coeffs_256,
                        filt_256,
                        dst,
                        dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                do {
                    convolve_x_16x2_8tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_x_16x2_8tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_x_16x2_8tap_avx2(
                        src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_x_16x2_8tap_avx2(
                        src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_x_16x2_8tap_avx2(
                        src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    convolve_x_16x2_8tap_avx2(
                        src_ptr + 64, 16, coeffs_256, filt_256, dst + 64, 16);
                    convolve_x_16x2_8tap_avx2(
                        src_ptr + 96, 16, coeffs_256, filt_256, dst + 96, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
    }
}
