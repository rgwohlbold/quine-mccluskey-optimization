#pragma once

#include "../../bitmap.h"
#include <assert.h>
#include <immintrin.h>

#ifndef __BMI2__
#error "need BMI2 as a base case"
#endif

#include "pext.h"

static inline void merge_avx2_single_register(int bit_difference, __m256i impl1, __m256i merged1, __m128i *result, __m256i *merged_result) {
    assert(0 <= bit_difference && bit_difference <= 7);

    int block_len = 1 << bit_difference;

    if (block_len == 128) {
        // set upper half to zero, lower half to upper half of impl1
        __m256i impl1_shuffled = _mm256_permute2x128_si256(impl1, impl1, 0x81);
        __m256i aggregated = _mm256_and_si256(impl1, impl1_shuffled);
        // set upper and lower halves to lower half of aggregated
        __m256i merged2 = _mm256_permute2x128_si256(aggregated, aggregated, 0x00);
        // use lower half of aggregated
        *result = _mm256_castsi256_si128(aggregated);
        *merged_result = _mm256_or_si256(merged1, merged2);
        return;
    }
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    if (block_len == 64) {
        // we need to shift across 64-bit boundaries which needs an immediate value
        impl2 = _mm256_srli_si256(impl1, 8); // 8 bytes = 64 bits
    } else {
        impl2 = _mm256_srli_epi64(impl1, block_len);
    }

    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
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
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    }
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    if (block_len == 64) {
        initial_result = aggregated;
    }
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    __m256i merged2 = _mm256_or_si256(merged1, initial_result);
    // shift initial_result left by block_len
    __m256i merged3;
    if (block_len == 64) {
        merged3 = _mm256_slli_si256(initial_result, 8); // 8 bytes = 64 bits
    } else {
        merged3 = _mm256_slli_epi64(initial_result, block_len);
    }
    *merged_result = _mm256_or_si256(merged2, merged3);
}

static inline void merge_avx2(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits <= 7) {
        merge_pext(implicants, merged, input_index, output_index, num_bits, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 256) { // implicants do not fit into one register, and we use the largest register size
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                for (int k = 0; k < block_len; k += 256) {
                    __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
                    __m256i impl2 = _mm256_load_si256((__m256i*)(implicants.bits + idx2 / 8));
                    __m256i merged1 = _mm256_load_si256((__m256i*)(merged.bits + idx1 / 8));
                    __m256i merged2 = _mm256_load_si256((__m256i*)(merged.bits + idx2 / 8));
                    __m256i res = _mm256_and_si256(impl1, impl2);
                    __m256i merged1_ = _mm256_or_si256(merged1, res);
                    __m256i merged2_ = _mm256_or_si256(merged2, res);
                    _mm256_store_si256((__m256i*)(merged.bits + idx1 / 8), merged1_);
                    _mm256_store_si256((__m256i*)(merged.bits + idx2 / 8), merged2_);
                    if (i >= first_difference) {
                        _mm256_store_si256((__m256i*)(implicants.bits + o_idx / 8), res);
                        o_idx += 256;
                    }
                    idx1 += 256;
                    idx2 += 256;
                }
            }
        } else { // implicants that are compared fit into one 256-bit register, i.e. block_len <= 128
            for (int block = 0; block < num_blocks; block += 128 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
                __m256i merged1 = _mm256_load_si256((__m256i*)(merged.bits + idx1 / 8));
                __m128i impl_result;
                __m256i merged_result;
                merge_avx2_single_register(i, impl1, merged1, &impl_result, &merged_result);
                _mm256_store_si256((__m256i*)(merged.bits + idx1 / 8), merged_result);
                if (i >= first_difference) {
                    _mm_store_si128((__m128i*)(implicants.bits + o_idx / 8), impl_result);
                    o_idx += 128;
                }
                idx1 += 256;
            }
        }
    }
}
