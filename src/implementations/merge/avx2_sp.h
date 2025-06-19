#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "pext_sp.h"

static inline void merge_avx2_sp_single_register(int bit_difference, __m256i impl1, __m256i primes1, __m128i *result,
                                                 __m256i *primes_result) {
    assert(0 <= bit_difference && bit_difference <= 7);

    int block_len = 1 << bit_difference;

    if (block_len == 128) {
        // set upper half to zero, lower half to upper half of impl1
        __m256i impl1_shuffled = _mm256_permute2x128_si256(impl1, impl1, 0x81);
        __m256i aggregated = _mm256_and_si256(impl1, impl1_shuffled);
        // set upper and lower halves to lower half of aggregated
        __m256i merged = _mm256_permute2x128_si256(aggregated, aggregated, 0x00);
        // use lower half of aggregated
        *result = _mm256_castsi256_si128(aggregated);
        *primes_result = _mm256_andnot_si256(merged, primes1);
        return;
    }
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    if (block_len == 64) {
        // we need to shift across 64-bit boundaries which needs an immediate value
        impl2 = _mm256_srli_si256(impl1, 8);  // 8 bytes = 64 bits
    } else {
        impl2 = _mm256_srli_epi64(impl1, block_len);
    }

    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_undefined_si256();  // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);
    if (block_len == 1) {
        aggregated = _mm256_and_si256(aggregated, _mm256_set1_epi8(0b01010101));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 1);
    }
    if (block_len <= 2) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00110011));
        if (block_len == 2) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 2);
    }
    if (block_len <= 4) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
        if (block_len == 4) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 4);
    }
    if (block_len <= 8) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
        if (block_len == 8) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 8);
    }
    if (block_len <= 16) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        if (block_len == 16) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 16);
    }
    if (block_len <= 32) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        if (block_len == 32) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_si256(aggregated, 4);  // 4 bytes = 32 bits
    }
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted),
                                  _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    if (block_len == 64) {
        initial_result = aggregated;
    }
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    if (block_len == 64) {
        shifted_initial_result = _mm256_slli_si256(initial_result, 8);  // 8 bytes = 64 bits
    } else {
        shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    }
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static void merge_avx2_sp(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                          int first_difference) {
    if (num_bits <= 7) {
        merge_pext_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 256) {  // implicants do not fit into one register, and we use the largest register size
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
        } else {  // implicants that are compared fit into one 256-bit register, i.e. block_len <= 128
            for (int block = 0; block < num_blocks; block += 128 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
                __m256i primes1;
                // assume we always run block_len=1 first and primes[idx1/8] is uninitialized at this point.
                // in that case, assume all implicants are prime
                if (block_len == 1) {
                    primes1 = impl1;
                } else {
                    primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
                }
                __m128i impl_result = _mm_undefined_si128();       // prevent uninitialized warnings
                __m256i primes_result = _mm256_undefined_si256();  // prevent uninitialized warnings
                merge_avx2_sp_single_register(i, impl1, primes1, &impl_result, &primes_result);
                _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes_result);
                if (i >= first_difference) {
                    _mm_store_si128((__m128i *)(implicants.bits + o_idx / 8), impl_result);
                    o_idx += 128;
                }
                idx1 += 256;
            }
        }
    }
}
