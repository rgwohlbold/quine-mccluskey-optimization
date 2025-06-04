#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "../../debug.h"
#include "./avx2_sp_richard.h"
#include "bits_sp.h"

/**
 * Uses Richard's single_register's implementation, but aims to achieve ILP through unrolling
 *
 */
static void merge_avx2_sp_aljaz(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                                int first_difference) {
    if (num_bits <= 7) {
        merge_avx2_sp_small_n_richard(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    int num_registers = (1 << num_bits) / 256;
    if (num_registers < 4) {
        // Less than 4 registers case, we do each register in a single pass
        for (int register_index = 0; register_index < num_registers; register_index += 1) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1 = o_idx + 128 * register_index;
            size_t o_idx1_b = o_idx1 >> 3;  // Divide by 8 to get byte index

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;

            // Expanded from for(i=0; i < 8; i++) { ...
            // with if(i >= first_difference) { ... }}

            __m128i impl_result[8] = {_mm_set1_epi64x(0), _mm_set1_epi64x(0), _mm_set1_epi64x(0), _mm_set1_epi64x(0),
                                      _mm_set1_epi64x(0), _mm_set1_epi64x(0), _mm_set1_epi64x(0), _mm_set1_epi64x(0)};
            __m256i primes_result[8] = {_mm256_set1_epi16(0), _mm256_set1_epi16(0), _mm256_set1_epi16(0),
                                        _mm256_set1_epi16(0), _mm256_set1_epi16(0), _mm256_set1_epi16(0),
                                        _mm256_set1_epi16(0), _mm256_set1_epi16(0)};

            merge_avx2_sp_single_register_richard_1(impl1, primes1, &impl_result[0], &primes_result[0]);
            merge_avx2_sp_single_register_richard_2(impl1, primes1, &impl_result[1], &primes_result[1]);
            merge_avx2_sp_single_register_richard_4(impl1, primes1, &impl_result[2], &primes_result[2]);
            merge_avx2_sp_single_register_richard_8(impl1, primes1, &impl_result[3], &primes_result[3]);
            merge_avx2_sp_single_register_richard_16(impl1, primes1, &impl_result[4], &primes_result[4]);
            merge_avx2_sp_single_register_richard_32(impl1, primes1, &impl_result[5], &primes_result[5]);
            merge_avx2_sp_single_register_richard_64(impl1, primes1, &impl_result[6], &primes_result[6]);
            merge_avx2_sp_single_register_richard_128(impl1, primes1, &impl_result[7], &primes_result[7]);

#define STORE_MERGED_AT_OFFSET(offset, result_idx) \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + offset * (num_registers * 16))), impl_result[result_idx])

            switch (first_difference) {
                case 0:
                    STORE_MERGED_AT_OFFSET(0, 0);
                    STORE_MERGED_AT_OFFSET(1, 1);
                    STORE_MERGED_AT_OFFSET(2, 2);
                    STORE_MERGED_AT_OFFSET(3, 3);
                    STORE_MERGED_AT_OFFSET(4, 4);
                    STORE_MERGED_AT_OFFSET(5, 5);
                    STORE_MERGED_AT_OFFSET(6, 6);
                    STORE_MERGED_AT_OFFSET(7, 7);
                    break;
                case 1:
                    STORE_MERGED_AT_OFFSET(0, 1);
                    STORE_MERGED_AT_OFFSET(1, 2);
                    STORE_MERGED_AT_OFFSET(2, 3);
                    STORE_MERGED_AT_OFFSET(3, 4);
                    STORE_MERGED_AT_OFFSET(4, 5);
                    STORE_MERGED_AT_OFFSET(5, 6);
                    STORE_MERGED_AT_OFFSET(6, 7);
                    break;
                case 2:
                    STORE_MERGED_AT_OFFSET(0, 2);
                    STORE_MERGED_AT_OFFSET(1, 3);
                    STORE_MERGED_AT_OFFSET(2, 4);
                    STORE_MERGED_AT_OFFSET(3, 5);
                    STORE_MERGED_AT_OFFSET(4, 6);
                    STORE_MERGED_AT_OFFSET(5, 7);
                    break;
                case 3:
                    STORE_MERGED_AT_OFFSET(0, 3);
                    STORE_MERGED_AT_OFFSET(1, 4);
                    STORE_MERGED_AT_OFFSET(2, 5);
                    STORE_MERGED_AT_OFFSET(3, 6);
                    STORE_MERGED_AT_OFFSET(4, 7);
                    break;
                case 4:
                    STORE_MERGED_AT_OFFSET(0, 4);
                    STORE_MERGED_AT_OFFSET(1, 5);
                    STORE_MERGED_AT_OFFSET(2, 6);
                    STORE_MERGED_AT_OFFSET(3, 7);
                    break;
                case 5:
                    STORE_MERGED_AT_OFFSET(0, 5);
                    STORE_MERGED_AT_OFFSET(1, 6);
                    STORE_MERGED_AT_OFFSET(2, 7);
                    break;
                case 6:
                    STORE_MERGED_AT_OFFSET(0, 6);
                    STORE_MERGED_AT_OFFSET(1, 7);
                    break;
                case 7:
                    STORE_MERGED_AT_OFFSET(0, 7);
                    break;
                default:
                    // No action needed for cases beyond 7
                    break;
            }

            // Tree-like gather the results
            primes_result[0] = _mm256_and_si256(primes_result[0], primes_result[1]);
            primes_result[2] = _mm256_and_si256(primes_result[2], primes_result[3]);
            primes_result[4] = _mm256_and_si256(primes_result[4], primes_result[5]);
            primes_result[6] = _mm256_and_si256(primes_result[6], primes_result[7]);
            primes_result[0] = _mm256_and_si256(primes_result[0], primes_result[2]);
            primes_result[4] = _mm256_and_si256(primes_result[4], primes_result[6]);
            primes1 = _mm256_and_si256(primes_result[0], primes_result[4]);
            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1);
        }
        if (first_difference <= 8) {
            o_idx += (8 - first_difference) * num_registers * 128;
        }
    } else {
        // More than or equal 4 registers case
        for (int register_index = 0; register_index < num_registers; register_index += 4) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1 = o_idx + 128 * register_index;
            size_t o_idx1_b = o_idx1 >> 3;  // Divide by 8 to get byte index

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;
            __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 256) / 8));
            __m256i primes2 = impl2;
            __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 512) / 8));
            __m256i primes3 = impl3;
            __m256i impl4 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 768) / 8));
            __m256i primes4 = impl4;

            // Expanded from for(i=0; i < 8; i++) { ...
            // with if(i >= first_difference) { ... write 4 ... }}

            __m128i impl_result[4][8];
            __m256i primes_result[4][8];
            merge_avx2_sp_single_register_richard_1(impl1, primes1, &impl_result[0][0], &primes_result[0][0]);
            merge_avx2_sp_single_register_richard_1(impl2, primes2, &impl_result[1][0], &primes_result[1][0]);
            merge_avx2_sp_single_register_richard_1(impl3, primes3, &impl_result[2][0], &primes_result[2][0]);
            merge_avx2_sp_single_register_richard_1(impl4, primes4, &impl_result[3][0], &primes_result[3][0]);

            merge_avx2_sp_single_register_richard_2(impl1, primes1, &impl_result[0][1], &primes_result[0][1]);
            merge_avx2_sp_single_register_richard_2(impl2, primes2, &impl_result[1][1], &primes_result[1][1]);
            merge_avx2_sp_single_register_richard_2(impl3, primes3, &impl_result[2][1], &primes_result[2][1]);
            merge_avx2_sp_single_register_richard_2(impl4, primes4, &impl_result[3][1], &primes_result[3][1]);

            merge_avx2_sp_single_register_richard_4(impl1, primes1, &impl_result[0][2], &primes_result[0][2]);
            merge_avx2_sp_single_register_richard_4(impl2, primes2, &impl_result[1][2], &primes_result[1][2]);
            merge_avx2_sp_single_register_richard_4(impl3, primes3, &impl_result[2][2], &primes_result[2][2]);
            merge_avx2_sp_single_register_richard_4(impl4, primes4, &impl_result[3][2], &primes_result[3][2]);

            merge_avx2_sp_single_register_richard_8(impl1, primes1, &impl_result[0][3], &primes_result[0][3]);
            merge_avx2_sp_single_register_richard_8(impl2, primes2, &impl_result[1][3], &primes_result[1][3]);
            merge_avx2_sp_single_register_richard_8(impl3, primes3, &impl_result[2][3], &primes_result[2][3]);
            merge_avx2_sp_single_register_richard_8(impl4, primes4, &impl_result[3][3], &primes_result[3][3]);

            merge_avx2_sp_single_register_richard_16(impl1, primes1, &impl_result[0][4], &primes_result[0][4]);
            merge_avx2_sp_single_register_richard_16(impl2, primes2, &impl_result[1][4], &primes_result[1][4]);
            merge_avx2_sp_single_register_richard_16(impl3, primes3, &impl_result[2][4], &primes_result[2][4]);
            merge_avx2_sp_single_register_richard_16(impl4, primes4, &impl_result[3][4], &primes_result[3][4]);

            merge_avx2_sp_single_register_richard_32(impl1, primes1, &impl_result[0][5], &primes_result[0][5]);
            merge_avx2_sp_single_register_richard_32(impl2, primes2, &impl_result[1][5], &primes_result[1][5]);
            merge_avx2_sp_single_register_richard_32(impl3, primes3, &impl_result[2][5], &primes_result[2][5]);
            merge_avx2_sp_single_register_richard_32(impl4, primes4, &impl_result[3][5], &primes_result[3][5]);

            merge_avx2_sp_single_register_richard_64(impl1, primes1, &impl_result[0][6], &primes_result[0][6]);
            merge_avx2_sp_single_register_richard_64(impl2, primes2, &impl_result[1][6], &primes_result[1][6]);
            merge_avx2_sp_single_register_richard_64(impl3, primes3, &impl_result[2][6], &primes_result[2][6]);
            merge_avx2_sp_single_register_richard_64(impl4, primes4, &impl_result[3][6], &primes_result[3][6]);

            merge_avx2_sp_single_register_richard_128(impl1, primes1, &impl_result[0][7], &primes_result[0][7]);
            merge_avx2_sp_single_register_richard_128(impl2, primes2, &impl_result[1][7], &primes_result[1][7]);
            merge_avx2_sp_single_register_richard_128(impl3, primes3, &impl_result[2][7], &primes_result[2][7]);
            merge_avx2_sp_single_register_richard_128(impl4, primes4, &impl_result[3][7], &primes_result[3][7]);

#define STORE_MERGED_QUAD_AT_OFFSET(offset, result_idx)                                             \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 0 + offset * (num_registers * 16))),  \
                    impl_result[0][result_idx]);                                                    \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16 + offset * (num_registers * 16))), \
                    impl_result[1][result_idx]);                                                    \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32 + offset * (num_registers * 16))), \
                    impl_result[2][result_idx]);                                                    \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48 + offset * (num_registers * 16))), \
                    impl_result[3][result_idx]);

            switch (first_difference) {
                case 0:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 0);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 1);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 2);
                    STORE_MERGED_QUAD_AT_OFFSET(3, 3);
                    STORE_MERGED_QUAD_AT_OFFSET(4, 4);
                    STORE_MERGED_QUAD_AT_OFFSET(5, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(6, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(7, 7);
                    break;
                case 1:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 1);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 2);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 3);
                    STORE_MERGED_QUAD_AT_OFFSET(3, 4);
                    STORE_MERGED_QUAD_AT_OFFSET(4, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(5, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(6, 7);
                    break;
                case 2:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 2);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 3);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 4);
                    STORE_MERGED_QUAD_AT_OFFSET(3, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(4, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(5, 7);
                    break;
                case 3:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 3);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 4);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(3, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(4, 7);
                    break;
                case 4:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 4);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(3, 7);
                    break;
                case 5:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 5);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(2, 7);
                    break;
                case 6:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 6);
                    STORE_MERGED_QUAD_AT_OFFSET(1, 7);
                    break;
                case 7:
                    STORE_MERGED_QUAD_AT_OFFSET(0, 7);
                    break;
                default:
                    // No action needed for cases beyond 7
                    break;
            }

            // After FOR
#define AND_8_TO_4(base_arr)                                  \
    base_arr[0] = _mm256_and_si256(base_arr[0], base_arr[1]); \
    base_arr[2] = _mm256_and_si256(base_arr[2], base_arr[3]); \
    base_arr[4] = _mm256_and_si256(base_arr[4], base_arr[5]); \
    base_arr[6] = _mm256_and_si256(base_arr[6], base_arr[7]);
#define AND_4_TO_2(base_arr)                                  \
    base_arr[0] = _mm256_and_si256(base_arr[0], base_arr[2]); \
    base_arr[4] = _mm256_and_si256(base_arr[4], base_arr[6]);
#define AND_2_TO_1(base_arr) base_arr[0] = _mm256_and_si256(base_arr[0], base_arr[4]);

            // Tree-like gather the results
            AND_8_TO_4(primes_result[0]);
            AND_8_TO_4(primes_result[1]);
            AND_8_TO_4(primes_result[2]);
            AND_8_TO_4(primes_result[3]);

            AND_4_TO_2(primes_result[0]);
            AND_4_TO_2(primes_result[1]);
            AND_4_TO_2(primes_result[2]);
            AND_4_TO_2(primes_result[3]);

            AND_2_TO_1(primes_result[0]);
            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes_result[0][0]);

            AND_2_TO_1(primes_result[1]);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 256) / 8), primes_result[1][0]);

            AND_2_TO_1(primes_result[2]);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 512) / 8), primes_result[2][0]);

            AND_2_TO_1(primes_result[3]);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 768) / 8), primes_result[3][0]);
        }
        if (first_difference <= 8) {
            o_idx += (8 - first_difference) * num_registers * 128;
        }
    }

    for (int i = 8; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            for (int k = 0; k < block_len; k += 256) {
                __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
                __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2 / 8));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2 / 8));
                __m256i res = _mm256_and_si256(impl1, impl2);
                __m256i primes1_ = _mm256_andnot_si256(res, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res, primes2);
                _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1_);
                _mm256_store_si256((__m256i *)(primes.bits + idx2 / 8), primes2_);
                if (i >= first_difference) {
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res);
                    o_idx += 256;
                }
                idx1 += 256;
                idx2 += 256;
            }
        }
    }
}
