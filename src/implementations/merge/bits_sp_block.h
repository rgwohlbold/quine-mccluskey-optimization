#pragma once

#include "../../bitmap.h"
#include "bits_sp.h"
#include "../../debug.h"



static inline void merge_bits_sp_block(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits == 1) {
        merge_bits_sp1(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 2) {
        merge_bits_sp2(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 3) {
        merge_bits_sp3(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 4) {
        merge_bits_sp4(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 5) {
        merge_bits_sp5(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 6) {
        merge_bits_sp6(implicants, primes, input_index, output_index, first_difference);
        return;
    }

    size_t o_idx1 = output_index;
    size_t o_idx2 = output_index;
    if (0 >= first_difference) { o_idx2 += (1 << (num_bits - 1)); }
    size_t o_idx4 = o_idx2;
    if (1 >= first_difference) { o_idx4 += (1 << (num_bits - 1)); }
    size_t o_idx8 = o_idx4;
    if (2 >= first_difference) { o_idx8 += (1 << (num_bits - 1)); }
    size_t o_idx16 = o_idx8;
    if (3 >= first_difference) { o_idx16 += (1 << (num_bits - 1)); }
    size_t o_idx32 = o_idx16;
    if (4 >= first_difference) { o_idx32 += (1 << (num_bits - 1)); }




    size_t num_registers = (1 << num_bits) / 64;
    for (size_t regist = 0; regist < num_registers; regist++) {
        // LOG_DEBUG("Block: %d, block_len: %d", block, block_len);
        size_t idx1 = input_index + regist * 64;
        uint64_t *input_ptr = (uint64_t *) implicants.bits;
        uint32_t *output_ptr = (uint32_t *) implicants.bits;
        uint64_t *primes_ptr = (uint64_t *) primes.bits;
        uint64_t original_implicant = input_ptr[idx1 / 64];
        uint64_t primes = original_implicant;
        
        uint64_t aggregated1 = original_implicant & (original_implicant >> 1);
        uint64_t aggregated2 = original_implicant & (original_implicant >> 2);
        uint64_t aggregated4 = original_implicant & (original_implicant >> 4);
        uint64_t aggregated8 = original_implicant & (original_implicant >> 8);
        uint64_t aggregated16 = original_implicant & (original_implicant >> 16);
        uint64_t aggregated32 = original_implicant & (original_implicant >> 32);
        
        uint64_t initial_result1 = 0;
        uint64_t initial_result2 = 0;
        uint64_t initial_result4 = 0;
        uint64_t initial_result8 = 0;
        uint64_t initial_result16 = 0;
        uint64_t initial_result32 = 0;

        uint64_t shifted1 = 0;
        uint64_t shifted2 = 0;
        uint64_t shifted4 = 0;
        uint64_t shifted8 = 0;
        uint64_t shifted16 = 0;

        aggregated1 = aggregated1 & 0b0101010101010101010101010101010101010101010101010101010101010101;
        initial_result1 = aggregated1;
        shifted1 = aggregated1 >> 1;

        aggregated1 = (aggregated1 | shifted1) & 0b0011001100110011001100110011001100110011001100110011001100110011;
        aggregated2 = aggregated2 & 0b0011001100110011001100110011001100110011001100110011001100110011;
        initial_result2 = aggregated2;
        shifted1 = aggregated1 >> 2;
        shifted2 = aggregated2 >> 2;

        aggregated1 = (aggregated1 | shifted1) & 0x0F0F0F0F0F0F0F0F;
        aggregated2 = (aggregated2 | shifted2) & 0x0F0F0F0F0F0F0F0F;
        aggregated4 = aggregated4 & 0x0F0F0F0F0F0F0F0F;
        initial_result4 = aggregated4;
        shifted1 = aggregated1 >> 4;
        shifted2 = aggregated2 >> 4;
        shifted4 = aggregated4 >> 4;

        aggregated1 = (aggregated1 | shifted1) & 0x00FF00FF00FF00FF;
        aggregated2 = (aggregated2 | shifted2) & 0x00FF00FF00FF00FF;
        aggregated4 = (aggregated4 | shifted4) & 0x00FF00FF00FF00FF;
        aggregated8 = aggregated8 & 0x00FF00FF00FF00FF;
        initial_result8 = aggregated8;
        shifted1 = aggregated1 >> 8;
        shifted2 = aggregated2 >> 8;
        shifted4 = aggregated4 >> 8;
        shifted8 = aggregated8 >> 8;
       
        aggregated1 = (aggregated1 | shifted1) & 0x0000FFFF0000FFFF;
        aggregated2 = (aggregated2 | shifted2) & 0x0000FFFF0000FFFF;
        aggregated4 = (aggregated4 | shifted4) & 0x0000FFFF0000FFFF;
        aggregated8 = (aggregated8 | shifted8) & 0x0000FFFF0000FFFF;
        aggregated16 = aggregated16 & 0x0000FFFF0000FFFF;
        initial_result16 = aggregated16;
        shifted1 = aggregated1 >> 16;
        shifted2 = aggregated2 >> 16;
        shifted4 = aggregated4 >> 16;
        shifted8 = aggregated8 >> 16;
        shifted16 = aggregated16 >> 16;
        
        aggregated1 = (aggregated1 | shifted1) & 0x00000000FFFFFFFF;
        aggregated2 = (aggregated2 | shifted2) & 0x00000000FFFFFFFF;
        aggregated4 = (aggregated4 | shifted4) & 0x00000000FFFFFFFF;
        aggregated8 = (aggregated8 | shifted8) & 0x00000000FFFFFFFF;
        aggregated16 = (aggregated16 | shifted16) & 0x00000000FFFFFFFF;
        aggregated32 = aggregated32 & 0x00000000FFFFFFFF;
        initial_result32 = aggregated32;
   

        uint64_t primes1 = primes & ~(initial_result1 | (initial_result1 << 1));
        uint64_t primes2 = primes1 & ~(initial_result2 | (initial_result2 << 2));
        uint64_t primes4 = primes2 & ~(initial_result4 | (initial_result4 << 4));
        uint64_t primes8 = primes4 & ~(initial_result8 | (initial_result8 << 8));
        uint64_t primes16 = primes8 & ~(initial_result16 | (initial_result16 << 16));
        uint64_t primes32 = primes16 & ~(initial_result32 | (initial_result32 << 32));

        primes_ptr[idx1 / 64] = primes32;
        if (0 >= first_difference) {
            output_ptr[o_idx1 / 32] = (uint32_t) aggregated1;
            o_idx1 += 32;
        }
        if (1 >= first_difference) {
            output_ptr[o_idx2 / 32] = (uint32_t) aggregated2;
            o_idx2 += 32;
        }
        if (2 >= first_difference) {
            output_ptr[o_idx4 / 32] = (uint32_t) aggregated4;
            o_idx4 += 32;
        }
        if (3 >= first_difference) {
            output_ptr[o_idx8 / 32] = (uint32_t) aggregated8;
            o_idx8 += 32;
        }
        if (4 >= first_difference) {
            output_ptr[o_idx16 / 32] = (uint32_t) aggregated16;
            o_idx16 += 32;
        }
        if (5 >= first_difference) {
            output_ptr[o_idx32 / 32] = (uint32_t) aggregated32;
            o_idx32 += 32;
        }
    }


    // normal
    size_t o_idx = o_idx32;
    for (int i = 6; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 64) { // implicants do not fit into one register, and we use the largest register size
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                for (int k = 0; k < block_len; k += 64) {
                    uint64_t *implicant_ptr = (uint64_t*) implicants.bits;
                    uint64_t *primes_ptr = (uint64_t*) primes.bits;

                    uint64_t impl1 = implicant_ptr[idx1 / 64];
                    uint64_t impl2 = implicant_ptr[idx2 / 64];
                    uint64_t primes1 = primes_ptr[idx1 / 64];
                    uint64_t primes2 = primes_ptr[idx2 / 64];
                    uint64_t res = impl1 & impl2;
                    uint64_t primes1_ = primes1 & ~res;
                    uint64_t primes2_ = primes2 & ~res;

                    primes_ptr[idx1 / 64] = primes1_;
                    primes_ptr[idx2 / 64] = primes2_;
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
                // LOG_DEBUG("Block: %d, block_len: %d", block, block_len);
                size_t idx1 = input_index + 2 * block * block_len;

                uint64_t *input_ptr = (uint64_t *) implicants.bits;
                uint32_t *output_ptr = (uint32_t *) implicants.bits;
                uint64_t *primes_ptr = (uint64_t *) primes.bits;
                uint64_t impl1 = input_ptr[idx1 / 64];
                uint64_t primes;
                if (block_len == 1) {
                    primes = impl1;
                }
                else {
                    primes = primes_ptr[idx1 / 64];
                }
                uint64_t impl2 = impl1 >> block_len;
                uint64_t aggregated = impl1 & impl2;
                uint64_t initial_result = 0;

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

                uint64_t primes2 = primes & ~(initial_result | (initial_result << block_len));

                primes_ptr[idx1 / 64] = primes2;
                if (i >= first_difference) {
                    output_ptr[o_idx / 32] = (uint32_t) aggregated;
                    o_idx += 32;
                }


            }
        }
    }
}

