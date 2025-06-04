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
        for (int register_index = 0; register_index < num_registers; register_index += 1) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1 = o_idx + 128 * register_index;
            size_t o_idx1_b = o_idx1 >> 3;  // Divide by 8 to get byte index

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;

            // Expanded from for(i=0; i < 8; i++) { ...
            // with if(i >= first_difference) { ... }}

            __m128i impl_result[8];
            __m256i primes_result[4];
            merge_avx2_sp_single_register_richard_1(impl1, primes1, &impl_result[0], &primes_result[0]);
            merge_avx2_sp_single_register_richard_2(impl1, primes1, &impl_result[1], &primes_result[1]);
            merge_avx2_sp_single_register_richard_4(impl1, primes1, &impl_result[2], &primes_result[2]);
            merge_avx2_sp_single_register_richard_8(impl1, primes1, &impl_result[3], &primes_result[3]);
            merge_avx2_sp_single_register_richard_16(impl1, primes_result[0], &impl_result[4], &primes_result[0]);
            merge_avx2_sp_single_register_richard_32(impl1, primes_result[1], &impl_result[5], &primes_result[1]);
            merge_avx2_sp_single_register_richard_64(impl1, primes_result[2], &impl_result[6], &primes_result[2]);
            merge_avx2_sp_single_register_richard_128(impl1, primes_result[3], &impl_result[7], &primes_result[3]);

#define STORE_MERGED_AT_OFFSET(offset, result_idx) \
    _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + offset * (num_registers << 4))), impl_result[result_idx])

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
            primes_result[0] = _mm256_and_si256(primes_result[0], primes_result[2]);

            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes_result[0]);
        }
        if (first_difference <= 8) {
            o_idx += (8 - first_difference) * num_registers * 128;
        }
    } else {
        for (int register_index = 0; register_index < num_registers; register_index += 4) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 16 * register_index;

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;
            __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 256) / 8));
            __m256i primes2 = impl2;
            __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 512) / 8));
            __m256i primes3 = impl3;
            __m256i impl4 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 768) / 8));
            __m256i primes4 = impl4;

            __m128i impl1_result, impl2_result, impl3_result, impl4_result;
            // __m256i primes1_result, primes2_result, primes3_result, primes4_result;

            merge_avx2_sp_single_register_richard_1(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_1(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_1(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_1(impl4, primes4, &impl4_result, &primes4);
            if (0 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_2(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_2(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_2(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_2(impl4, primes4, &impl4_result, &primes4);
            if (1 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_4(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_4(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_4(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_4(impl4, primes4, &impl4_result, &primes4);
            if (2 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_8(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_8(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_8(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_8(impl4, primes4, &impl4_result, &primes4);
            if (3 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_16(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_16(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_16(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_16(impl4, primes4, &impl4_result, &primes4);
            if (4 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_32(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_32(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_32(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_32(impl4, primes4, &impl4_result, &primes4);
            if (5 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_64(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_64(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_64(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_64(impl4, primes4, &impl4_result, &primes4);
            if (6 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_richard_128(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_richard_128(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_richard_128(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_richard_128(impl4, primes4, &impl4_result, &primes4);
            if (7 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 256) / 8), primes2);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 512) / 8), primes3);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 768) / 8), primes4);
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
