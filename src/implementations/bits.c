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

static void merge_implicants_bits(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants that are compared fit into one register
        if (num_bits >= 6 && block_len <= 32) {
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
        } else {
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                if (block_len >= 64) {
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
                } else if (block_len >= 8) {
                    for (int k = 0; k < block_len; k += 8) {
                        uint8_t impl1 = implicants.bits[idx1 / 8];
                        uint8_t impl2 = implicants.bits[idx2 / 8];
                        uint8_t merged1 = merged.bits[idx1 / 8];
                        uint8_t merged2 = merged.bits[idx2 / 8];
                        uint8_t res = impl1 & impl2;
                        uint8_t merged1_ = merged1 | res;
                        uint8_t merged2_ = merged2 | res;

                        merged.bits[idx1 / 8] = merged1_;
                        merged.bits[idx2 / 8] = merged2_;
                        if (i >= first_difference) {
                            implicants.bits[o_idx / 8] = res;
                            o_idx += 8;
                        }
                        idx1 += 8;
                        idx2 += 8;
                    }
                } else {
                    for (int k = 0; k < block_len; k++) {
                        bool impl1 = BITMAP_CHECK(implicants, idx1);
                        bool impl2 = BITMAP_CHECK(implicants, idx2);
                        bool merged1 = BITMAP_CHECK(merged, idx1);
                        bool merged2 = BITMAP_CHECK(merged, idx2);
                        bool res = impl1 && impl2;
                        bool merged1_ = merged1 || res;
                        bool merged2_ = merged2 || res;

                        BITMAP_SET(merged, idx1, merged1_);
                        BITMAP_SET(merged, idx2, merged2_);

                        if (i >= first_difference) {
                            BITMAP_SET(implicants, o_idx, res);
                            o_idx++;
                        }
                        idx1++;
                        idx2++;
                    }
                }
            }
        }
    }
}

prime_implicant_result prime_implicants_bits(int num_bits, int num_trues, int *trues) {
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
            merge_implicants_bits(implicants, merged, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }

#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    // Step 2: Scan for unmerged implicants
    for (size_t i = 0; i < num_implicants / 64; i++) {
        uint64_t implicant_true = ((uint64_t*)implicants.bits)[i];
        uint64_t merged_true = ((uint64_t*)merged.bits)[i];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t*)primes.bits)[i] = prime_true;
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