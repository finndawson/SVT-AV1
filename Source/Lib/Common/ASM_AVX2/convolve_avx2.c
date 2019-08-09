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
    uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h,
    InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn, const int32_t subpel_y_qn,
    ConvolveParams *conv_params) {
    int32_t i, j;
    // right shift is F-1 because we are already dividing
    // filter co-efficients by 2
    const int32_t right_shift_bits = (FILTER_BITS - 1);
    const __m128i right_shift = _mm_cvtsi32_si128(right_shift_bits);
    const __m256i right_shift_const =
        _mm256_set1_epi16((1 << right_shift_bits) >> 1);

    assert(conv_params->round_0 <= FILTER_BITS);
    assert(((conv_params->round_0 + conv_params->round_1) <= (FILTER_BITS + 1)) ||
        ((conv_params->round_0 + conv_params->round_1) == (2 * FILTER_BITS)));

    (void)filter_params_x;
    (void)subpel_x_qn;
    (void)conv_params;
    __m256i coeffs[4], s[8];
    __m128i d[6];

    if (is_convolve_2tap(filter_params_y->filter_ptr)) {
        // vert_filt as 2 tap
        const int32_t fo_vert = 0;
        const uint8_t *const src_ptr = src - fo_vert * src_stride;

        prepare_coeffs_lowbd_2tap_avx2(filter_params_y, subpel_y_qn, coeffs);

        for (j = 0; j < w; j += 16) {
            const uint8_t *data = &src_ptr[j];
            d[0] = _mm_loadu_si128((__m128i *)(data + 0 * src_stride));

            for (i = 0; i < h; i += 2) {
                data = &src_ptr[i * src_stride + j];
                d[1] = _mm_loadu_si128((__m128i *)(data + 1 * src_stride));
                const __m256i src_45a = _mm256_permute2x128_si256(
                    _mm256_castsi128_si256(d[0]), _mm256_castsi128_si256(d[1]), 0x20);

                d[0] = _mm_loadu_si128((__m128i *)(data + 2 * src_stride));
                const __m256i src_56a = _mm256_permute2x128_si256(
                    _mm256_castsi128_si256(d[1]), _mm256_castsi128_si256(d[0]), 0x20);

                s[0] = _mm256_unpacklo_epi8(src_45a, src_56a);
                s[1] = _mm256_unpackhi_epi8(src_45a, src_56a);

                const __m256i res_lo = convolve_lowbd_2tap_avx2(s, coeffs);
                /* rounding code */
                // shift by F - 1
                const __m256i res_16b_lo = _mm256_sra_epi16(
                    _mm256_add_epi16(res_lo, right_shift_const), right_shift);
                // 8 bit conversion and saturation to uint8
                __m256i res_8b_lo = _mm256_packus_epi16(res_16b_lo, res_16b_lo);

                if (w - j > 8) {
                    const __m256i res_hi = convolve_lowbd_2tap_avx2(s + 1, coeffs);

                    /* rounding code */
                    // shift by F - 1
                    const __m256i res_16b_hi = _mm256_sra_epi16(
                        _mm256_add_epi16(res_hi, right_shift_const), right_shift);
                    // 8 bit conversion and saturation to uint8
                    __m256i res_8b_hi = _mm256_packus_epi16(res_16b_hi, res_16b_hi);

                    __m256i res_a = _mm256_unpacklo_epi64(res_8b_lo, res_8b_hi);

                    const __m128i res_0 = _mm256_castsi256_si128(res_a);
                    const __m128i res_1 = _mm256_extracti128_si256(res_a, 1);

                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res_0);
                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j + dst_stride],
                        res_1);
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_8b_lo);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8b_lo, 1);
                    if (w - j > 4) {
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res_0);
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j + dst_stride],
                            res_1);
                    }
                    else if (w - j > 2) {
                        xx_storel_32(&dst[i * dst_stride + j], res_0);
                        xx_storel_32(&dst[i * dst_stride + j + dst_stride], res_1);
                    }
                    else {
                        __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];
                        __m128i *const p_1 =
                            (__m128i *)&dst[i * dst_stride + j + dst_stride];
                        *(uint16_t *)p_0 = _mm_cvtsi128_si32(res_0);
                        *(uint16_t *)p_1 = _mm_cvtsi128_si32(res_1);
                    }
                }
            }
        }
    }
    else if (is_convolve_4tap(filter_params_y->filter_ptr)) {
        // vert_filt as 4 tap
        const int32_t fo_vert = 1;
        const uint8_t *const src_ptr = src - fo_vert * src_stride;

        prepare_coeffs_lowbd_8tap_avx2(filter_params_y, subpel_y_qn, coeffs);

        for (j = 0; j < w; j += 16) {
            const uint8_t *data = &src_ptr[j];
            d[0] = _mm_loadu_si128((__m128i *)(data + 0 * src_stride));
            d[1] = _mm_loadu_si128((__m128i *)(data + 1 * src_stride));
            d[2] = _mm_loadu_si128((__m128i *)(data + 2 * src_stride));
            d[3] = _mm_loadu_si128((__m128i *)(data + 3 * src_stride));
            d[4] = _mm_loadu_si128((__m128i *)(data + 4 * src_stride));

            // Load lines a and b. Line a to lower 128, line b to upper 128
            const __m256i src_01a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[0]), _mm256_castsi128_si256(d[1]), 0x20);

            const __m256i src_12a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[1]), _mm256_castsi128_si256(d[2]), 0x20);

            const __m256i src_23a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[2]), _mm256_castsi128_si256(d[3]), 0x20);

            const __m256i src_34a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[3]), _mm256_castsi128_si256(d[4]), 0x20);

            s[0] = _mm256_unpacklo_epi8(src_01a, src_12a);
            s[1] = _mm256_unpacklo_epi8(src_23a, src_34a);

            s[3] = _mm256_unpackhi_epi8(src_01a, src_12a);
            s[4] = _mm256_unpackhi_epi8(src_23a, src_34a);

            for (i = 0; i < h; i += 2) {
                data = &src_ptr[i * src_stride + j];
                d[5] = _mm_loadu_si128((__m128i *)(data + 5 * src_stride));
                const __m256i src_45a = _mm256_permute2x128_si256(
                    _mm256_castsi128_si256(d[4]), _mm256_castsi128_si256(d[5]), 0x20);

                d[4] = _mm_loadu_si128((__m128i *)(data + 6 * src_stride));
                const __m256i src_56a = _mm256_permute2x128_si256(
                    _mm256_castsi128_si256(d[5]), _mm256_castsi128_si256(d[4]), 0x20);

                s[2] = _mm256_unpacklo_epi8(src_45a, src_56a);
                s[5] = _mm256_unpackhi_epi8(src_45a, src_56a);

                const __m256i res_lo = convolve_lowbd_4tap_avx2(s, coeffs + 1);
                /* rounding code */
                // shift by F - 1
                const __m256i res_16b_lo = _mm256_sra_epi16(
                    _mm256_add_epi16(res_lo, right_shift_const), right_shift);
                // 8 bit conversion and saturation to uint8
                __m256i res_8b_lo = _mm256_packus_epi16(res_16b_lo, res_16b_lo);

                if (w - j > 8) {
                    const __m256i res_hi = convolve_lowbd_4tap_avx2(s + 3, coeffs + 1);

                    /* rounding code */
                    // shift by F - 1
                    const __m256i res_16b_hi = _mm256_sra_epi16(
                        _mm256_add_epi16(res_hi, right_shift_const), right_shift);
                    // 8 bit conversion and saturation to uint8
                    __m256i res_8b_hi = _mm256_packus_epi16(res_16b_hi, res_16b_hi);

                    __m256i res_a = _mm256_unpacklo_epi64(res_8b_lo, res_8b_hi);

                    const __m128i res_0 = _mm256_castsi256_si128(res_a);
                    const __m128i res_1 = _mm256_extracti128_si256(res_a, 1);

                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res_0);
                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j + dst_stride],
                        res_1);
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_8b_lo);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8b_lo, 1);
                    if (w - j > 4) {
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res_0);
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j + dst_stride],
                            res_1);
                    }
                    else if (w - j > 2) {
                        xx_storel_32(&dst[i * dst_stride + j], res_0);
                        xx_storel_32(&dst[i * dst_stride + j + dst_stride], res_1);
                    }
                    else {
                        __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];
                        __m128i *const p_1 =
                            (__m128i *)&dst[i * dst_stride + j + dst_stride];
                        *(uint16_t *)p_0 = _mm_cvtsi128_si32(res_0);
                        *(uint16_t *)p_1 = _mm_cvtsi128_si32(res_1);
                    }
                }
                s[0] = s[1];
                s[1] = s[2];

                s[3] = s[4];
                s[4] = s[5];
            }
        }
    }
    else {
        const int32_t fo_vert = filter_params_y->taps / 2 - 1;
        const uint8_t *const src_ptr = src - fo_vert * src_stride;

        prepare_coeffs_lowbd_8tap_avx2(filter_params_y, subpel_y_qn, coeffs);

        for (j = 0; j < w; j += 16) {
            const uint8_t *data = &src_ptr[j];
            __m256i src6;

            d[0] = _mm_loadu_si128((__m128i *)(data + 0 * src_stride));
            d[1] = _mm_loadu_si128((__m128i *)(data + 1 * src_stride));
            d[2] = _mm_loadu_si128((__m128i *)(data + 2 * src_stride));
            d[3] = _mm_loadu_si128((__m128i *)(data + 3 * src_stride));
            d[4] = _mm_loadu_si128((__m128i *)(data + 4 * src_stride));
            d[5] = _mm_loadu_si128((__m128i *)(data + 5 * src_stride));
            // Load lines a and b. Line a to lower 128, line b to upper 128
            const __m256i src_01a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[0]), _mm256_castsi128_si256(d[1]), 0x20);

            const __m256i src_12a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[1]), _mm256_castsi128_si256(d[2]), 0x20);

            const __m256i src_23a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[2]), _mm256_castsi128_si256(d[3]), 0x20);

            const __m256i src_34a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[3]), _mm256_castsi128_si256(d[4]), 0x20);

            const __m256i src_45a = _mm256_permute2x128_si256(
                _mm256_castsi128_si256(d[4]), _mm256_castsi128_si256(d[5]), 0x20);

            src6 = _mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *)(data + 6 * src_stride)));
            const __m256i src_56a =
                _mm256_permute2x128_si256(_mm256_castsi128_si256(d[5]), src6, 0x20);

            s[0] = _mm256_unpacklo_epi8(src_01a, src_12a);
            s[1] = _mm256_unpacklo_epi8(src_23a, src_34a);
            s[2] = _mm256_unpacklo_epi8(src_45a, src_56a);

            s[4] = _mm256_unpackhi_epi8(src_01a, src_12a);
            s[5] = _mm256_unpackhi_epi8(src_23a, src_34a);
            s[6] = _mm256_unpackhi_epi8(src_45a, src_56a);

            for (i = 0; i < h; i += 2) {
                data = &src_ptr[i * src_stride + j];
                const __m256i src_67a = _mm256_permute2x128_si256(
                    src6,
                    _mm256_castsi128_si256(
                        _mm_loadu_si128((__m128i *)(data + 7 * src_stride))),
                    0x20);

                src6 = _mm256_castsi128_si256(
                    _mm_loadu_si128((__m128i *)(data + 8 * src_stride)));
                const __m256i src_78a = _mm256_permute2x128_si256(
                    _mm256_castsi128_si256(
                        _mm_loadu_si128((__m128i *)(data + 7 * src_stride))),
                    src6, 0x20);

                s[3] = _mm256_unpacklo_epi8(src_67a, src_78a);
                s[7] = _mm256_unpackhi_epi8(src_67a, src_78a);

                const __m256i res_lo = convolve_lowbd_8tap_avx2(s, coeffs);

                /* rounding code */
                // shift by F - 1
                const __m256i res_16b_lo = _mm256_sra_epi16(
                    _mm256_add_epi16(res_lo, right_shift_const), right_shift);
                // 8 bit conversion and saturation to uint8
                __m256i res_8b_lo = _mm256_packus_epi16(res_16b_lo, res_16b_lo);

                if (w - j > 8) {
                    const __m256i res_hi = convolve_lowbd_8tap_avx2(s + 4, coeffs);

                    /* rounding code */
                    // shift by F - 1
                    const __m256i res_16b_hi = _mm256_sra_epi16(
                        _mm256_add_epi16(res_hi, right_shift_const), right_shift);
                    // 8 bit conversion and saturation to uint8
                    __m256i res_8b_hi = _mm256_packus_epi16(res_16b_hi, res_16b_hi);

                    __m256i res_a = _mm256_unpacklo_epi64(res_8b_lo, res_8b_hi);

                    const __m128i res_0 = _mm256_castsi256_si128(res_a);
                    const __m128i res_1 = _mm256_extracti128_si256(res_a, 1);

                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res_0);
                    _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j + dst_stride],
                        res_1);
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_8b_lo);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8b_lo, 1);
                    if (w - j > 4) {
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res_0);
                        _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j + dst_stride],
                            res_1);
                    }
                    else if (w - j > 2) {
                        xx_storel_32(&dst[i * dst_stride + j], res_0);
                        xx_storel_32(&dst[i * dst_stride + j + dst_stride], res_1);
                    }
                    else {
                        __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];
                        __m128i *const p_1 =
                            (__m128i *)&dst[i * dst_stride + j + dst_stride];
                        *(uint16_t *)p_0 = _mm_cvtsi128_si32(res_0);
                        *(uint16_t *)p_1 = _mm_cvtsi128_si32(res_1);
                    }
                }
                s[0] = s[1];
                s[1] = s[2];
                s[2] = s[3];

                s[4] = s[5];
                s[5] = s[6];
                s[6] = s[7];
            }
        }
    }
}

void eb_av1_convolve_x_sr_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h,
    InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn, const int32_t subpel_y_qn,
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
                prepare_coeffs_lowbd_2tap_ssse3(filter_params_x, subpel_x_qn, coeffs_128);

                if (w == 2) {
                    const __m128i c = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);

                    do {
                        __m128i s;

                        s = _mm_cvtsi32_si128(*(const int32_t *)src_ptr);
                        s = _mm_insert_epi32(s, *(int32_t *)(src_ptr + src_stride), 2);
                        s = _mm_shuffle_epi8(s, c);
                        const __m128i d = convolve_lowbd_x_8_2tap_kernel_ssse3(s, coeffs_128);
                        *(uint16_t *)dst = (uint16_t)_mm_cvtsi128_si32(d);
                        *(uint16_t *)(dst + dst_stride) = _mm_extract_epi16(d, 2);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 4) {
                    const __m128i c = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);

                    do {
                        __m128i s;

                        s = _mm_loadl_epi64((__m128i *)src_ptr);
                        s = _mm_loadh_epi64(src_ptr + src_stride, s);
                        s = _mm_shuffle_epi8(s, c);
                        const __m128i d = convolve_lowbd_x_8_2tap_kernel_ssse3(s, coeffs_128);
                        xx_storel_32(dst, d);
                        *(int32_t *)(dst + dst_stride) = _mm_extract_epi32(d, 1);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else {
                    assert(w == 8);

                    do {
                        __m128i s[2];

                        const __m128i s00 = _mm_loadu_si128((__m128i *)src_ptr);
                        const __m128i s10 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                        const __m128i s01 = _mm_srli_si128(s00, 1);
                        const __m128i s11 = _mm_srli_si128(s10, 1);
                        s[0] = _mm_unpacklo_epi8(s00, s01);
                        s[1] = _mm_unpacklo_epi8(s10, s11);
                        __m128i res_16b[2];

                        res_16b[0] = convolve_lowbd_2tap_ssse3(&s[0], coeffs_128);
                        res_16b[1] = convolve_lowbd_2tap_ssse3(&s[1], coeffs_128);
                        res_16b[0] = convolve_round_sse2(res_16b[0]);
                        res_16b[1] = convolve_round_sse2(res_16b[1]);
                        const __m128i d = _mm_packus_epi16(res_16b[0], res_16b[1]);
                        _mm_storel_epi64((__m128i *)dst, d);
                        _mm_storeh_epi64((__m128i *)(dst + dst_stride), d);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
            }
            else {
                prepare_coeffs_lowbd_2tap_avx2(filter_params_x, subpel_x_qn, coeffs_256);

                if (w == 16) {
                    do {
                        __m128i s_128[2][2];
                        __m256i s[2];

                        s_128[0][0] = _mm_loadu_si128((__m128i *)src_ptr);
                        s_128[0][1] = _mm_loadu_si128((__m128i *)(src_ptr + 1));
                        s_128[1][0] = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                        s_128[1][1] = _mm_loadu_si128((__m128i *)(src_ptr + src_stride + 1));
                        s[0] = _mm256_setr_m128i(s_128[0][0], s_128[1][0]);
                        s[1] = _mm256_setr_m128i(s_128[0][1], s_128[1][1]);
                        const __m256i d = convolve_lowbd_x_32_2tap_kernel_avx2(s, coeffs_256);
                        const __m128i d0 = _mm256_castsi256_si128(d);
                        const __m128i d1 = _mm256_extracti128_si256(d, 1);
                        _mm_storeu_si128((__m128i *)dst, d0);
                        _mm_storeu_si128((__m128i *)(dst + dst_stride), d1);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        i -= 2;
                    } while (i);
                }
                else if (w == 32) {
                    do {
                        convolve_lowbd_x_32_2tap_avx2(src_ptr, coeffs_256, dst);
                        src_ptr += src_stride;
                        dst += dst_stride;
                    } while (--i);
                }
                else if (w == 64) {
                    do {
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 0 * 32, coeffs_256, dst + 0 * 32);
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 1 * 32, coeffs_256, dst + 1 * 32);
                        src_ptr += src_stride;
                        dst += dst_stride;
                    } while (--i);
                }
                else {
                    assert(w == 128);

                    do {
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 0 * 32, coeffs_256, dst + 0 * 32);
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 1 * 32, coeffs_256, dst + 1 * 32);
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 2 * 32, coeffs_256, dst + 2 * 32);
                        convolve_lowbd_x_32_2tap_avx2(src_ptr + 3 * 32, coeffs_256, dst + 3 * 32);
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
                    __m128i s;

                    s = _mm_cvtsi32_si128(*(const int32_t *)src_ptr);
                    s = _mm_insert_epi32(s, *(int32_t *)(src_ptr + src_stride), 2);
                    const __m128i s1 = _mm_srli_si128(s, 1);
                    const __m128i d = _mm_avg_epu8(s, s1);
                    *(uint16_t *)dst = (uint16_t)_mm_cvtsi128_si32(d);
                    *(uint16_t *)(dst + dst_stride) = _mm_extract_epi16(d, 4);

                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 4) {
                do {
                    __m128i s;

                    s = _mm_loadl_epi64((__m128i *)src_ptr);
                    s = _mm_loadh_epi64(src_ptr + src_stride, s);
                    const __m128i s1 = _mm_srli_si128(s, 1);
                    const __m128i d = _mm_avg_epu8(s, s1);
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
                    const __m128i s10 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
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
                    const __m128i s01 = _mm_loadu_si128((__m128i *)(src_ptr + 1));
                    const __m128i s10 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m128i s11 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride + 1));
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
                    convolve_lowbd_x_32_2tap_avg(src_ptr, dst);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 0 * 32, dst + 0 * 32);
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 1 * 32, dst + 1 * 32);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 0 * 32, dst + 0 * 32);
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 1 * 32, dst + 1 * 32);
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 2 * 32, dst + 2 * 32);
                    convolve_lowbd_x_32_2tap_avg(src_ptr + 3 * 32, dst + 3 * 32);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
    }
    else if (is_convolve_4tap(filter_params_x->filter_ptr)) {
        // horz_filt as 4 tap
        const int32_t fo_horiz = 1;
        const uint8_t *src_ptr = src - fo_horiz;
        const __m128i c0 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12);
        const __m128i c1 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 10, 11, 11, 12, 12, 13, 13, 14);
        __m128i t, s[2];

        prepare_coeffs_lowbd_4tap_ssse3(filter_params_x, subpel_x_qn, coeffs_128);

        if (w == 2) {
            do {
                t = _mm_loadl_epi64((__m128i *)src_ptr);
                t = _mm_loadh_epi64(src_ptr + src_stride, t);
                s[0] = _mm_shuffle_epi8(t, c0);
                s[1] = _mm_shuffle_epi8(t, c1);
                const __m128i d = convolve_lowbd_x_8_4tap_kernel_ssse3(s, coeffs_128);
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
                s[0] = _mm_shuffle_epi8(t, c0);
                s[1] = _mm_shuffle_epi8(t, c1);
                const __m128i d = convolve_lowbd_x_8_4tap_kernel_ssse3(s, coeffs_128);
                xx_storel_32(dst, d);
                *(int32_t *)(dst + dst_stride) = _mm_extract_epi32(d, 1);

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
            const int32_t fo_horiz = 2;
            const uint8_t *src_ptr = src - fo_horiz;

            prepare_coeffs_lowbd_6tap_avx2(filter_params_x, subpel_x_qn, coeffs_256);

            if (w == 8) {
                do {
                    const __m128i s0 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s1 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m256i s = _mm256_setr_m128i(s0, s1);
                    const __m256i res0 = convolve_lowbd_x_6tap_avx2(s, coeffs_256, filt_256);
                    const __m256i res1 = convolve_round_avx2(res0);
                    const __m256i d = _mm256_packus_epi16(res1, res1);
                    const __m128i d0 = _mm256_castsi256_si128(d);
                    const __m128i d1 = _mm256_extracti128_si256(d, 1);
                    _mm_storel_epi64((__m128i *)dst, d0);
                    _mm_storel_epi64((__m128i *)(dst + dst_stride), d1);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                do {
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr, src_stride, coeffs_256, filt_256, dst, dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                do {
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr + 64, 16, coeffs_256, filt_256, dst + 64, 16);
                    convolve_lowbd_x_16x2_6tap_avx2(src_ptr + 96, 16, coeffs_256, filt_256, dst + 96, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
        else {
            // horz_filt as 8 tap
            const int32_t fo_horiz = 3;
            const uint8_t *src_ptr = src - fo_horiz;

            filt_256[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

            prepare_coeffs_lowbd_8tap_avx2(filter_params_x, subpel_x_qn, coeffs_256);

            if (w == 8) {
                do {
                    const __m128i s0 = _mm_loadu_si128((__m128i *)src_ptr);
                    const __m128i s1 = _mm_loadu_si128((__m128i *)(src_ptr + src_stride));
                    const __m256i s = _mm256_setr_m128i(s0, s1);
                    const __m256i res0 = convolve_lowbd_x_8tap_avx2(s, coeffs_256, filt_256);
                    const __m256i res1 = convolve_round_avx2(res0);
                    const __m256i d = _mm256_packus_epi16(res1, res1);
                    const __m128i d0 = _mm256_castsi256_si128(d);
                    const __m128i d1 = _mm256_extracti128_si256(d, 1);
                    _mm_storel_epi64((__m128i *)dst, d0);
                    _mm_storel_epi64((__m128i *)(dst + dst_stride), d1);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 16) {
                do {
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr, src_stride, coeffs_256, filt_256, dst, dst_stride);
                    src_ptr += 2 * src_stride;
                    dst += 2 * dst_stride;
                    i -= 2;
                } while (i);
            }
            else if (w == 32) {
                do {
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else if (w == 64) {
                do {
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
            else {
                assert(w == 128);

                do {
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr, 16, coeffs_256, filt_256, dst, 16);
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr + 32, 16, coeffs_256, filt_256, dst + 32, 16);
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr + 64, 16, coeffs_256, filt_256, dst + 64, 16);
                    convolve_lowbd_x_16x2_8tap_avx2(src_ptr + 96, 16, coeffs_256, filt_256, dst + 96, 16);
                    src_ptr += src_stride;
                    dst += dst_stride;
                } while (--i);
            }
        }
    }
}
