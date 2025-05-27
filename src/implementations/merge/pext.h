#pragma once

#include "../../bitmap.h"
#include "bits.h"
#include <immintrin.h>

static inline void merge_pext(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits == 1) {
        merge_bits1(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 2) {
        merge_bits2(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 3) {
        merge_bits3(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 4) {
        merge_bits4(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 5) {
        merge_bits5(implicants, merged, input_index, output_index, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 64) { // implicants do not fit into one register, and we use the largest register size
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

                    uint64_t mask;
                    if (block_len == 1) {
                        mask = 0b0101010101010101010101010101010101010101010101010101010101010101;
                    } else if (block_len == 2) {
                        mask = 0b0011001100110011001100110011001100110011001100110011001100110011;
                    } else if (block_len == 4) {
                        mask = 0x0F0F0F0F0F0F0F0F;
                    } else if (block_len == 8) {
                        mask = 0x00FF00FF00FF00FF;
                    } else if (block_len == 16) {
                        mask = 0x0000FFFF0000FFFF;
                    } else { // block_len == 32
                        mask = 0x00000000FFFFFFFF;
                    }
                    uint64_t result = _pext_u64(aggregated, mask);
                    uint64_t initial_result = aggregated & mask;

                    uint64_t merged2 = merged | initial_result | (initial_result << block_len);

                    merged_ptr[idx1 / 64] = merged2;
                    if (i >= first_difference) {
                        output_ptr[o_idx / 32] = (uint32_t) result;
                        o_idx += 32;
                    }
                    idx1 += 64;
                }
            }
        }
    }
}
