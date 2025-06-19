#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "../../debug.h"
#include "./avx2_sp_ssa.h"
#include "bits_sp.h"

static void merge_avx2_sp_unroll(bitmap implicants, bitmap primes, size_t input_index, size_t output_index,
                                 int num_bits, int first_difference) {
    if (num_bits <= 7) {
        merge_avx2_sp_small_n_ssa(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    int num_registers = (1 << num_bits) / 256;
    if (num_registers < 4) {
        for (int register_index = 0; register_index < num_registers; register_index += 1) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 16 * register_index;

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;
            for (int i = 0; i < 8; i++) {
                __m128i impl_result = _mm_undefined_si128();       // prevent uninitialized warnings
                __m256i primes_result = _mm256_undefined_si256();  // prevent uninitialized warnings

                merge_avx2_sp_single_register_ssa(i, impl1, primes1, &impl_result, &primes_result);
                primes1 = primes_result;
                if (i >= first_difference) {
                    _mm_store_si128((__m128i *)(implicants.bits + o_idx1_b), impl_result);
                    o_idx1_b += 16 * num_registers;
                }
            }
            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1);
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

            merge_avx2_sp_single_register_ssa_1(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_1(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_1(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_1(impl4, primes4, &impl4_result, &primes4);
            if (0 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_2(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_2(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_2(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_2(impl4, primes4, &impl4_result, &primes4);
            if (1 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_4(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_4(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_4(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_4(impl4, primes4, &impl4_result, &primes4);
            if (2 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_8(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_8(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_8(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_8(impl4, primes4, &impl4_result, &primes4);
            if (3 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_16(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_16(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_16(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_16(impl4, primes4, &impl4_result, &primes4);
            if (4 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_32(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_32(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_32(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_32(impl4, primes4, &impl4_result, &primes4);
            if (5 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_64(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_64(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_64(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_64(impl4, primes4, &impl4_result, &primes4);
            if (6 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_ssa_128(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_ssa_128(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_ssa_128(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_ssa_128(impl4, primes4, &impl4_result, &primes4);
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
