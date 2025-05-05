#ifdef __AVX2__
#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../debug.h"
#include "bits.h"
#include "immintrin.h"

void merge_implicants_avx2(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits <= 7) {
        merge_implicants_bits(implicants, merged, input_index, output_index, num_bits, first_difference);
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
                    uint64_t *implicant_ptr = (uint64_t*) implicants.bits;
                    uint64_t *merged_ptr = (uint64_t*) merged.bits;

                    __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
                    __m256i impl2 = _mm256_load_si256((__m256i*)(implicants.bits + idx2 / 8));
                    __m256i merged1 = _mm256_load_si256((__m256i*)(merged.bits + idx1 / 8));
                    __m256i merged2 = _mm256_load_si256((__m256i*)(merged.bits + idx2 / 8));
                    __m256i res = _mm256_and_si256(impl1, impl2);
                    __m256i merged1_ = _mm256_or_si256(merged1, res);
                    __m256i merged2_ = _mm256_or_si256(merged2, res);
                    //_mm256_store_si256((__m256i*)(merged.bits + idx1 / 8), merged1_);
                    //_mm256_store_si256((__m256i*)(merged.bits + idx2 / 8), merged2_);
                    if (i >= first_difference) {
                        //_mm256_store_si256((__m256i*)(implicants.bits + o_idx / 8), res);
                        o_idx += 256;
                    }
                    idx1 += 256;
                    idx2 += 256;
                }
            }
        } else if (block_len >= 64) { // implicants do not fit into one register, and we use the largest register size
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                for (int k = 0; k < block_len; k += 64) {
                    uint64_t *implicant_ptr = (uint64_t*) implicants.bits;
                    uint64_t *merged_ptr = (uint64_t*) merged.bits;

                    uint64_t impl1 = implicant_ptr[idx1 / 64];
                    uint64_t impl2 = implicant_ptr[idx2 / 64];
                    uint64_t merged1 = merged_ptr[idx1 / 64];
                    uint64_t merged2 = merged_ptr[idx2 / 64];
                    uint64_t res = impl1 & impl2;
                    uint64_t merged1_ = merged1 | res;
                    uint64_t merged2_ = merged2 | res;

                    merged_ptr[idx1 / 64] = merged1_;
                    merged_ptr[idx2 / 64] = merged2_;
                    if (i >= first_difference) {
                        implicant_ptr[o_idx / 64] = res;
                        o_idx += 64;
                    }
                    idx1 += 64;
                    idx2 += 64;
                }
            }
        } else { // implicants that are compared fit into one 64-bit register
            for (int block = 0; block < num_blocks; block += 32 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                uint64_t *input_ptr = (uint64_t *) implicants.bits;
                uint32_t *output_ptr = (uint32_t *) implicants.bits;
                uint64_t *merged_ptr = (uint64_t *) merged.bits;
                for (int k = 0; k < block_len; k += 64) {
                    uint64_t impl1 = input_ptr[idx1 / 64];
                    uint64_t merged = merged_ptr[idx1 / 64];

                    uint64_t impl2 = impl1 >> block_len;
                    uint64_t aggregated = impl1 & impl2;

                    uint64_t initial_result;

                    uint64_t shifted = 0;
                    if (block_len == 1) {
                        aggregated = aggregated & 0b0101010101010101010101010101010101010101010101010101010101010101;
                        initial_result = aggregated;
                        shifted = aggregated >> 1;
                    }
                    if (block_len <= 2) {
                        aggregated = (aggregated | shifted) & 0b0011001100110011001100110011001100110011001100110011001100110011;
                        if (block_len == 2) {
                            initial_result = aggregated;
                        }
                        shifted = aggregated >> 2;
                    }
                    if (block_len <= 4) {
                        aggregated = (aggregated | shifted) & 0x0F0F0F0F0F0F0F0F;
                        if (block_len == 4) {
                            initial_result = aggregated;
                        }
                        shifted = aggregated >> 4;
                    }
                    if (block_len <= 8) {
                        aggregated = (aggregated | shifted) & 0x00FF00FF00FF00FF;
                        if (block_len == 8) {
                            initial_result = aggregated;
                        }
                        shifted = aggregated >> 8;
                    }
                    if (block_len <= 16) {
                        aggregated = (aggregated | shifted) & 0x0000FFFF0000FFFF;
                        if (block_len == 16) {
                            initial_result = aggregated;
                        }
                        shifted = aggregated >> 16;
                    }
                    aggregated = (aggregated | shifted) & 0x00000000FFFFFFFF;
                    if (block_len == 32) {
                        initial_result = aggregated;
                    }

                    uint64_t merged2 = merged | initial_result | (initial_result << block_len);

                    merged_ptr[idx1 / 64] = merged2;
                    if (i >= first_difference) {
                        output_ptr[o_idx / 32] = (uint32_t) aggregated;
                        o_idx += 32;
                    }
                    idx1 += 64;
                }
            }
        } /*else { // implicants that are compared fit into one 256-bit register
            for (int block = 0; block < num_blocks; block += 128 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                for (int k = 0; k < block_len; k += 256) {
                    __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
                    __m256i merged1 = _mm256_load_si256((__m256i*)(merged.bits + idx1 / 8));

                    // we would like to shift impl1 right by block_len bits. However, we cannot shift across 128-bit lanes.
                    // therefore, use a high and a low part and combine them to simulate that shift
                    __m256i lo = _mm256_srli_si256(impl1, 1); // TODO
                    __m256i hi = _mm256_slli_si256(impl1, 128 - 1); // TODO
                    __m256i hi_shuffled = _mm256_permute2x128_si256(hi, hi, 0x81);
                    __m256i impl2 = _mm256_or_si256(lo, hi_shuffled);

                    __m256i aggregated = _mm256_and_si256(impl1, impl2);
                    __m256i initial_result;

                    __m256i shifted;
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
                        shifted = _mm256_srli_epi64(aggregated, 32);
                    }
                    if (block_len <= 64) {
                        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
                        if (block_len == 64) {
                            initial_result = aggregated;
                        }
                    }
                    aggregated = _mm256_permute4x64_epi64(aggregated, 0x88);
                    __m128i result = _mm256_castsi256_si128(aggregated);

                    __m256i merged2 = _mm256_or_si256(merged1, initial_result);

                    // now we need to shift initial_result left across 128-bit lanes
                    __m256i shifted_low = _mm256_srli_si256(initial_result, 128 - 1); // TODO
                    __m256i shifted_high = _mm256_slli_si256(initial_result, 1); // TODO
                    __m256i cross_bits = _mm256_permute2x128_si256(shifted_low, shifted_low, 0x08);
                    __m256i shifted_initial_result = _mm256_or_si256(shifted_high, cross_bits);

                    __m256i merged3 = _mm256_or_si256(merged2, shifted_initial_result);

                    _mm256_store_epi64(merged.bits + idx1 / 8, merged3);
                    if (i >= first_difference) {
                        _mm_store_epi64(implicants.bits + o_idx / 8, result);
                        o_idx += 128;
                    }
                    idx1 += 256;
                }
            }
        } */
    }
}

prime_implicant_result prime_implicants_avx2(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }
    bitmap merged = bitmap_allocate(num_implicants);

    uint64_t num_ops = 0;
    init_tsc();
    uint64_t counter_start = start_tsc();

    size_t input_index = 0;
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            merge_implicants_avx2(implicants, merged, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }

#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    // Step 2: Scan for unmerged implicants
    for (size_t i = 0; i < num_implicants - (num_implicants % 256); i += 256) {
        __m256i implicant_true = _mm256_load_si256((__m256i*)(implicants.bits + i / 8));
        __m256i merged_true = _mm256_load_si256((__m256i*)(merged.bits + i / 8));
        __m256i prime_true = _mm256_andnot_si256(merged_true, implicant_true);
        _mm256_store_si256((__m256i*)(primes.bits + i / 8), prime_true);
    }
    for (size_t i = num_implicants - (num_implicants % 256); i < num_implicants - num_implicants % 64; i += 64) {
        uint64_t implicant_true = ((uint64_t*)implicants.bits)[i / 64];
        uint64_t merged_true = ((uint64_t*)merged.bits)[i / 64];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t*)primes.bits)[i / 64] = prime_true;
    }
    for (size_t i = num_implicants - (num_implicants % 64); i < num_implicants; i++) {
        if (BITMAP_CHECK(implicants, i) && !BITMAP_CHECK(merged, i)) {
            BITMAP_SET_TRUE(primes, i);
        }
    }
#ifdef COUNT_OPS
        num_ops += 2 * num_implicants;
#endif

    uint64_t cycles = stop_tsc(counter_start);
    bitmap_free(implicants);
    bitmap_free(merged);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
#ifdef COUNT_OPS
        .num_ops = num_ops,
#endif
    };
    return result;
}
#endif