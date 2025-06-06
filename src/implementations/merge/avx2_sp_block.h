#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "../../debug.h"
#include "./avx2_sp_shuffle.h"
#include "bits_sp.h"

static void merge_avx2_sp_block(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                                int first_difference) {
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
                __m128i impl_result;
                __m256i primes_result;

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

            merge_avx2_sp_single_register_shuffle_1(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_1(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_1(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_1(impl4, primes4, &impl4_result, &primes4);
            if (0 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_2(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_2(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_2(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_2(impl4, primes4, &impl4_result, &primes4);
            if (1 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_4(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_4(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_4(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_4(impl4, primes4, &impl4_result, &primes4);
            if (2 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_8(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_8(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_8(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_8(impl4, primes4, &impl4_result, &primes4);
            if (3 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_16(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_16(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_16(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_16(impl4, primes4, &impl4_result, &primes4);
            if (4 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_32(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_32(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_32(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_32(impl4, primes4, &impl4_result, &primes4);
            if (5 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_64(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_64(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_64(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_64(impl4, primes4, &impl4_result, &primes4);
            if (6 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_128(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_128(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_128(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_128(impl4, primes4, &impl4_result, &primes4);
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

    size_t input_index_b = input_index / 8;
    int i = 8;
    for (; i+2 < num_bits; i += 3) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 4) {
            size_t idx0 = input_index_b + (2 * block * block_len) / 8;
            size_t idx1 = input_index_b + (2 * block * block_len + block_len) / 8;
            size_t idx2 = input_index_b + (2 * block * block_len + 2 * block_len) / 8;
            size_t idx3 = input_index_b + (2 * block * block_len + 3 * block_len) / 8;
            size_t idx4 = input_index_b + (2 * block * block_len + 4 * block_len) / 8;
            size_t idx5 = input_index_b + (2 * block * block_len + 5 * block_len) / 8;
            size_t idx6 = input_index_b + (2 * block * block_len + 6 * block_len) / 8;
            size_t idx7 = input_index_b + (2 * block * block_len + 7 * block_len) / 8;

            for (int k = 0; k < block_len; k += 256) {
                __m256i impl0 = _mm256_load_si256((__m256i *)(implicants.bits + idx0));
                __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1));
                __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2));
                __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + idx3));
                __m256i impl4 = _mm256_load_si256((__m256i *)(implicants.bits + idx4));
                __m256i impl5 = _mm256_load_si256((__m256i *)(implicants.bits + idx5));
                __m256i impl6 = _mm256_load_si256((__m256i *)(implicants.bits + idx6));
                __m256i impl7 = _mm256_load_si256((__m256i *)(implicants.bits + idx7));

                __m256i primes0 = _mm256_load_si256((__m256i *)(primes.bits + idx0));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2));
                __m256i primes3 = _mm256_load_si256((__m256i *)(primes.bits + idx3));
                __m256i primes4 = _mm256_load_si256((__m256i *)(primes.bits + idx4));
                __m256i primes5 = _mm256_load_si256((__m256i *)(primes.bits + idx5));
                __m256i primes6 = _mm256_load_si256((__m256i *)(primes.bits + idx6));
                __m256i primes7 = _mm256_load_si256((__m256i *)(primes.bits + idx7));

                __m256i res01 = _mm256_and_si256(impl0, impl1);
                __m256i res23 = _mm256_and_si256(impl2, impl3);
                __m256i res45 = _mm256_and_si256(impl4, impl5);
                __m256i res67 = _mm256_and_si256(impl6, impl7);
                __m256i res02 = _mm256_and_si256(impl0, impl2);
                __m256i res13 = _mm256_and_si256(impl1, impl3);
                __m256i res46 = _mm256_and_si256(impl4, impl6);
                __m256i res57 = _mm256_and_si256(impl5, impl7);
                __m256i res04 = _mm256_and_si256(impl0, impl4);
                __m256i res15 = _mm256_and_si256(impl1, impl5);
                __m256i res26 = _mm256_and_si256(impl2, impl6);
                __m256i res37 = _mm256_and_si256(impl3, impl7);

                __m256i primes0_ = _mm256_andnot_si256(res01, primes0);
                __m256i primes1_ = _mm256_andnot_si256(res01, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res23, primes2);
                __m256i primes3_ = _mm256_andnot_si256(res23, primes3);
                __m256i primes4_ = _mm256_andnot_si256(res45, primes4);
                __m256i primes5_ = _mm256_andnot_si256(res45, primes5);
                __m256i primes6_ = _mm256_andnot_si256(res67, primes6);
                __m256i primes7_ = _mm256_andnot_si256(res67, primes7);

                __m256i primes0__ = _mm256_andnot_si256(res02, primes0_);
                __m256i primes1__ = _mm256_andnot_si256(res13, primes1_);
                __m256i primes2__ = _mm256_andnot_si256(res02, primes2_);
                __m256i primes3__ = _mm256_andnot_si256(res13, primes3_);
                __m256i primes4__ = _mm256_andnot_si256(res46, primes4_);
                __m256i primes5__ = _mm256_andnot_si256(res57, primes5_);
                __m256i primes6__ = _mm256_andnot_si256(res46, primes6_);
                __m256i primes7__ = _mm256_andnot_si256(res57, primes7_);

                __m256i primes0___ = _mm256_andnot_si256(res04, primes0__);
                __m256i primes1___ = _mm256_andnot_si256(res15, primes1__);
                __m256i primes2___ = _mm256_andnot_si256(res26, primes2__);
                __m256i primes3___ = _mm256_andnot_si256(res37, primes3__);
                __m256i primes4___ = _mm256_andnot_si256(res04, primes4__);
                __m256i primes5___ = _mm256_andnot_si256(res15, primes5__);
                __m256i primes6___ = _mm256_andnot_si256(res26, primes6__);
                __m256i primes7___ = _mm256_andnot_si256(res37, primes7__);

                _mm256_store_si256((__m256i *)(primes.bits + idx0), primes0___);
                _mm256_store_si256((__m256i *)(primes.bits + idx1), primes1___);
                _mm256_store_si256((__m256i *)(primes.bits + idx2), primes2___);
                _mm256_store_si256((__m256i *)(primes.bits + idx3), primes3___);
                _mm256_store_si256((__m256i *)(primes.bits + idx4), primes4___);
                _mm256_store_si256((__m256i *)(primes.bits + idx5), primes5___);
                _mm256_store_si256((__m256i *)(primes.bits + idx6), primes6___);
                _mm256_store_si256((__m256i *)(primes.bits + idx7), primes7___);
                if (i >= first_difference) {
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res01);
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + (block+1) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res23);
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + (block+2) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res45);
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + (block+3) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res67);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res02);
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + (block+1) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res13);
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + (block+2) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res46);
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + (block+3) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res57);
                }
                if (i+2 >= first_difference) {
                    o_idx = output_index + ((i+2 - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res04);
                    o_idx = output_index + ((i+2 - first_difference) << (num_bits - 1)) + (block+1) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res15);
                    o_idx = output_index + ((i+2 - first_difference) << (num_bits - 1)) + (block+2) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res26);
                    o_idx = output_index + ((i+2 - first_difference) << (num_bits - 1)) + (block+3) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res37);
                }
                idx0 += 32;
                idx1 += 32;
                idx2 += 32;
                idx3 += 32;
                idx4 += 32;
                idx5 += 32;
                idx6 += 32;
                idx7 += 32;
            }
        }
    }
    for (; i+1 < num_bits; i += 2) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 2) {
            size_t idx0 = input_index + 2 * block * block_len;
            size_t idx1 = input_index + 2 * block * block_len + block_len;
            size_t idx2 = input_index + 2 * block * block_len + 2 * block_len;
            size_t idx3 = input_index + 2 * block * block_len + 3 * block_len;

            for (int k = 0; k < block_len; k += 256) {
                __m256i impl0 = _mm256_load_si256((__m256i *)(implicants.bits + idx0 / 8));
                __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
                __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2 / 8));
                __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + idx3 / 8));

                __m256i primes0 = _mm256_load_si256((__m256i *)(primes.bits + idx0 / 8));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2 / 8));
                __m256i primes3 = _mm256_load_si256((__m256i *)(primes.bits + idx3 / 8));

                __m256i res01 = _mm256_and_si256(impl0, impl1);
                __m256i res23 = _mm256_and_si256(impl2, impl3);
                __m256i res02 = _mm256_and_si256(impl0, impl2);
                __m256i res13 = _mm256_and_si256(impl1, impl3);

                __m256i primes0_ = _mm256_andnot_si256(res01, primes0);
                __m256i primes1_ = _mm256_andnot_si256(res01, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res23, primes2);
                __m256i primes3_ = _mm256_andnot_si256(res23, primes3);

                __m256i primes0__ = _mm256_andnot_si256(res02, primes0_);
                __m256i primes1__ = _mm256_andnot_si256(res13, primes1_);
                __m256i primes2__ = _mm256_andnot_si256(res02, primes2_);
                __m256i primes3__ = _mm256_andnot_si256(res13, primes3_);

                _mm256_store_si256((__m256i *)(primes.bits + idx0 / 8), primes0__);
                _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1__);
                _mm256_store_si256((__m256i *)(primes.bits + idx2 / 8), primes2__);
                _mm256_store_si256((__m256i *)(primes.bits + idx3 / 8), primes3__);
                if (i >= first_difference) {
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res01);
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + (block+1) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res23);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res02);
                    o_idx = output_index + ((i+1 - first_difference) << (num_bits - 1)) + (block+1) * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res13);
                }
                idx0 += 256;
                idx1 += 256;
                idx2 += 256;
                idx3 += 256;
            }
        }
    }

    for (; i < num_bits; i++) {
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
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res);
                }
                idx1 += 256;
                idx2 += 256;
            }
        }
    }

    // we load 8 registers at the same time to do three block sizes at the same time
    //int i = 8;
    //for (; i + 1 < num_bits; i += 2) {
    //    int block_len = 1 << i;
    //    int num_blocks = 1 << (num_bits - i - 1);
    //    LOG_DEBUG("entering %d %d %d %lu", num_bits, block_len, num_blocks, o_idx);

    //    size_t o_idx1 = o_idx;
    //    if (i >= first_difference) {
    //        o_idx1 += 1 << (num_bits - 1);
    //    }

    //    // implicants do not fit into one register, and we use the largest register size
    //    for (int block = 0; block < num_blocks; block += 2) {
    //        size_t idx0 = input_index + 2 * block * block_len;
    //        size_t idx1 = input_index + 2 * block * block_len + block_len;
    //        size_t idx2 = input_index + 2 * block * block_len + 2 * block_len;
    //        size_t idx3 = input_index + 2 * block * block_len + 3 * block_len;

    //        for (int k = 0; k < block_len; k += 256) {
    //            LOG_DEBUG("block_len=%d num_blocks=%d idx0=%lu idx1=%lu idx2=%lu idx3=%lu", block_len, num_blocks, idx0, idx1, idx2, idx3);
    //            __m256i impl0 = _mm256_load_si256((__m256i *)(implicants.bits + idx0 / 8));
    //            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
    //            __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2 / 8));
    //            __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + idx3 / 8));
    //            //log_m256i("impl0", &impl0);
    //            //log_m256i("impl1", &impl1);
    //            //log_m256i("impl2", &impl2);
    //            //log_m256i("impl3", &impl3);
    //            __m256i primes0 = _mm256_load_si256((__m256i *)(primes.bits + idx0 / 8));
    //            __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
    //            __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2 / 8));
    //            __m256i primes3 = _mm256_load_si256((__m256i *)(primes.bits + idx3 / 8));

    //            __m256i res01 = _mm256_and_si256(impl0, impl1);
    //            __m256i res23 = _mm256_and_si256(impl2, impl3);
    //            __m256i res02 = _mm256_and_si256(impl0, impl2);
    //            __m256i res13 = _mm256_and_si256(impl1, impl3);

    //            //log_m256i("res01", &res01);
    //            //log_m256i("res23", &res23);
    //            //log_m256i("res02", &res02);
    //            //log_m256i("res13", &res13);

    //            __m256i primes0_ = _mm256_andnot_si256(res01, primes0);
    //            __m256i primes0__ = _mm256_andnot_si256(res02, primes0_);
    //            __m256i primes1_ = _mm256_andnot_si256(res01, primes1);
    //            __m256i primes1__ = _mm256_andnot_si256(res13, primes1_);
    //            __m256i primes2_ = _mm256_andnot_si256(res23, primes2);
    //            __m256i primes2__ = _mm256_andnot_si256(res02, primes2_);
    //            __m256i primes3_ = _mm256_andnot_si256(res23, primes3);
    //            __m256i primes3__ = _mm256_andnot_si256(res13, primes3_);

    //            _mm256_store_si256((__m256i *)(primes.bits + idx0 / 8), primes0__);
    //            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1__);
    //            _mm256_store_si256((__m256i *)(primes.bits + idx2 / 8), primes2__);
    //            _mm256_store_si256((__m256i *)(primes.bits + idx3 / 8), primes3__);

    //            LOG_DEBUG("storing primes at %lu", idx0);
    //            log_m256i("data: ", &primes0__);
    //            LOG_DEBUG("storing primes at %lu", idx1);
    //            log_m256i("data: ", &primes1__);
    //            LOG_DEBUG("storing primes at %lu", idx2);
    //            log_m256i("data: ", &primes2__);
    //            LOG_DEBUG("storing primes at %lu", idx3);
    //            log_m256i("data: ", &primes3__);

    //            if (i >= first_difference) {
    //                o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
    //                _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res01);
    //                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx+256) / 8), res23);
    //                LOG_DEBUG("writing %lu", o_idx)
    //                log_m256i("data:", &res01);
    //                LOG_DEBUG("writing %lu", o_idx+256)
    //                log_m256i("data:", &res23);
    //            }
    //            if (i+1 >= first_difference) {
    //                o_idx1 = output_index + ((i+1 - first_difference) << (num_bits - 1)) + block * block_len + k;
    //                _mm256_store_si256((__m256i *)(implicants.bits + o_idx1 / 8), res02); // chunk next to output chunk
    //                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1+256) / 8), res13); // chunk next to output chunk
    //                LOG_DEBUG("writing %lu", o_idx1)
    //                log_m256i("data:", &res02);
    //                LOG_DEBUG("writing %lu", o_idx1+256)
    //                log_m256i("data:", &res13);
    //            }
    //            idx0 += 256;
    //            idx1 += 256;
    //            idx2 += 256;
    //            idx3 += 256;
    //        }
    //    }
    //    o_idx = o_idx1;
    //}
    //// do remaining block size
    //for (; i < num_bits; i++) {
    //    int block_len = 1 << i;
    //    int num_blocks = 1 << (num_bits - i - 1);
    //    LOG_DEBUG("entering %d %d %d %lu", num_bits, block_len, num_blocks, o_idx);

    //    // implicants do not fit into one register, and we use the largest register size
    //    for (int block = 0; block < num_blocks; block++) {
    //        size_t idx1 = input_index + 2 * block * block_len;
    //        size_t idx2 = input_index + 2 * block * block_len + block_len;

    //        for (int k = 0; k < block_len; k += 256) {
    //            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
    //            __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2 / 8));
    //            __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
    //            __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2 / 8));
    //            __m256i res = _mm256_and_si256(impl1, impl2);
    //            __m256i primes1_ = _mm256_andnot_si256(res, primes1);
    //            __m256i primes2_ = _mm256_andnot_si256(res, primes2);
    //            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1_);
    //            _mm256_store_si256((__m256i *)(primes.bits + idx2 / 8), primes2_);
    //            LOG_DEBUG("storing primes at %lu", idx1);
    //            log_m256i("data: ", &primes1_);
    //            LOG_DEBUG("storing primes at %lu", idx2);
    //            log_m256i("data: ", &primes2_);
    //            if (i >= first_difference) {
    //                o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
    //                _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res);
    //                LOG_DEBUG("writing %lu", o_idx)
    //                log_m256i("data:", &res);
    //                o_idx += 256;
    //            }
    //            idx1 += 256;
    //            idx2 += 256;
    //        }
    //    }
    //}
}
