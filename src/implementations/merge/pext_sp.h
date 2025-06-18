#pragma once

#include <immintrin.h>

#include "../../bitmap.h"
#include "bits_sp.h"

static inline void merge_pext_sp(bitmap implicants, bitmap primes, size_t input_index, size_t output_index,
                                 int num_bits, int first_difference) {
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
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 64) {  // implicants do not fit into one register, and we use the largest register size
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                for (int k = 0; k < block_len; k += 64) {
                    uint64_t *implicant_ptr = (uint64_t *)implicants.bits;
                    uint64_t *primes_ptr = (uint64_t *)primes.bits;

                    uint64_t impl1 = implicant_ptr[idx1 / 64];
                    uint64_t impl2 = implicant_ptr[idx2 / 64];
                    uint64_t prime1 = primes_ptr[idx1 / 64];
                    uint64_t prime2 = primes_ptr[idx2 / 64];
                    uint64_t res = impl1 & impl2;
                    uint64_t prime1_ = prime1 & ~res;
                    uint64_t prime2_ = prime2 & ~res;

                    primes_ptr[idx1 / 64] = prime1_;
                    primes_ptr[idx2 / 64] = prime2_;
                    if (i >= first_difference) {
                        implicant_ptr[o_idx / 64] = res;
                        o_idx += 64;
                    }
                    idx1 += 64;
                    idx2 += 64;
                }
            }
        } else {  // implicants that are compared fit into one 64-bit register
            for (int block = 0; block < num_blocks; block += 32 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                uint64_t *input_ptr = (uint64_t *)implicants.bits;
                uint32_t *output_ptr = (uint32_t *)implicants.bits;
                uint64_t *primes_ptr = (uint64_t *)primes.bits;
                for (int k = 0; k < block_len; k += 64) {
                    uint64_t impl1 = input_ptr[idx1 / 64];
                    uint64_t prime;
                    if (block_len == 1) {
                        prime = impl1;
                    } else {
                        prime = primes_ptr[idx1 / 64];
                    }

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
                    } else {  // block_len == 32
                        mask = 0x00000000FFFFFFFF;
                    }
                    uint64_t result = _pext_u64(aggregated, mask);
                    uint64_t initial_result = aggregated & mask;

                    uint64_t prime2 = prime & ~(initial_result | (initial_result << block_len));

                    primes_ptr[idx1 / 64] = prime2;
                    if (i >= first_difference) {
                        output_ptr[o_idx / 32] = (uint32_t)result;
                        o_idx += 32;
                    }
                    idx1 += 64;
                }
            }
        }
    }
}
