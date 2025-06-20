#pragma once

#include <immintrin.h>

#include "../../bitmap.h"
#include "bits_sp.h"

static inline void merge_pext_sp_unroll_ilp(bitmap implicants, bitmap primes, size_t input_index, size_t output_index,
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
    const size_t input_index_64 = input_index / 64;    // input_index is in bits, convert to 64-bit words
    const size_t output_index_32 = output_index / 32;  // output_index is in bits, convert to 32-bit words
    // Layer size: 2^(num_bits - 1) /32 // Divided by 32 for index in 32-bit word writing
    size_t next_layer_size_w = (1 << (num_bits - 1 - 5));

    size_t o_idx1 = output_index_32;

    size_t o_idx2 = output_index_32;
    if (0 >= first_difference) {
        o_idx2 += next_layer_size_w;
    }
    size_t o_idx4 = o_idx2;
    if (1 >= first_difference) {
        o_idx4 += next_layer_size_w;
    }
    size_t o_idx8 = o_idx4;
    if (2 >= first_difference) {
        o_idx8 += next_layer_size_w;
    }
    size_t o_idx16 = o_idx8;
    if (3 >= first_difference) {
        o_idx16 += next_layer_size_w;
    }
    size_t o_idx32 = o_idx16;
    if (4 >= first_difference) {
        o_idx32 += next_layer_size_w;
    }

    size_t num_registers = (1 << num_bits) / 64;  // Number of 64-bit registers needed for the block
    uint64_t *implicants_ptr = (uint64_t *)implicants.bits;
    uint32_t *output_ptr = (uint32_t *)implicants.bits;
    uint64_t *primes_ptr = (uint64_t *)primes.bits;

#define MASK1 0b0101010101010101010101010101010101010101010101010101010101010101
#define MASK2 0b0011001100110011001100110011001100110011001100110011001100110011
#define MASK4 0x0F0F0F0F0F0F0F0F
#define MASK8 0x00FF00FF00FF00FF
#define MASK16 0x0000FFFF0000FFFF
#define MASK32 0x00000000FFFFFFFF

    if (num_registers < 4) {
        // We can't fully pipeline the operations, so we do one by one.
        for (size_t regist = 0; regist < num_registers; regist++) {
            // LOAD
            size_t idx1 = (input_index_64 + regist);
            uint64_t original_implicant = implicants_ptr[idx1];
            uint64_t primes = original_implicant;

            // COMPUTE
            uint64_t aggregated1 = original_implicant & (original_implicant >> 1);
            uint64_t aggregated2 = original_implicant & (original_implicant >> 2);
            uint64_t aggregated4 = original_implicant & (original_implicant >> 4);
            uint64_t aggregated8 = original_implicant & (original_implicant >> 8);
            uint64_t aggregated16 = original_implicant & (original_implicant >> 16);
            uint64_t aggregated32 = original_implicant & (original_implicant >> 32);

            uint64_t result1 = _pext_u64(aggregated1, MASK1);
            uint64_t result2 = _pext_u64(aggregated2, MASK2);
            uint64_t result4 = _pext_u64(aggregated4, MASK4);
            uint64_t result8 = _pext_u64(aggregated8, MASK8);
            uint64_t result16 = _pext_u64(aggregated16, MASK16);
            uint64_t result32 = _pext_u64(aggregated32, MASK32);

            uint64_t initial_result1 = aggregated1 & MASK1;
            uint64_t initial_result2 = aggregated2 & MASK2;
            uint64_t initial_result4 = aggregated4 & MASK4;
            uint64_t initial_result8 = aggregated8 & MASK8;
            uint64_t initial_result16 = aggregated16 & MASK16;
            uint64_t initial_result32 = aggregated32 & MASK32;

            uint64_t primes1 = primes & ~(initial_result1 | (initial_result1 << 1));
            uint64_t primes2 = primes1 & ~(initial_result2 | (initial_result2 << 2));
            uint64_t primes4 = primes2 & ~(initial_result4 | (initial_result4 << 4));
            uint64_t primes8 = primes4 & ~(initial_result8 | (initial_result8 << 8));
            uint64_t primes16 = primes8 & ~(initial_result16 | (initial_result16 << 16));
            uint64_t primes32 = primes16 & ~(initial_result32 | (initial_result32 << 32));

            // STORE
            primes_ptr[idx1] = primes32;
            if (0 == first_difference) {
                output_ptr[o_idx1++] = (uint32_t)result1;
                output_ptr[o_idx2++] = (uint32_t)result2;
                output_ptr[o_idx4++] = (uint32_t)result4;
                output_ptr[o_idx8++] = (uint32_t)result8;
                output_ptr[o_idx16++] = (uint32_t)result16;
                output_ptr[o_idx32++] = (uint32_t)result32;
            } else if (1 == first_difference) {
                output_ptr[o_idx2++] = (uint32_t)result2;
                output_ptr[o_idx4++] = (uint32_t)result4;
                output_ptr[o_idx8++] = (uint32_t)result8;
                output_ptr[o_idx16++] = (uint32_t)result16;
                output_ptr[o_idx32++] = (uint32_t)result32;
            } else if (2 == first_difference) {
                output_ptr[o_idx4++] = (uint32_t)result4;
                output_ptr[o_idx8++] = (uint32_t)result8;
                output_ptr[o_idx16++] = (uint32_t)result16;
                output_ptr[o_idx32++] = (uint32_t)result32;
            } else if (3 == first_difference) {
                output_ptr[o_idx8++] = (uint32_t)result8;
                output_ptr[o_idx16++] = (uint32_t)result16;
                output_ptr[o_idx32++] = (uint32_t)result32;
            } else if (4 == first_difference) {
                output_ptr[o_idx16++] = (uint32_t)result16;
                output_ptr[o_idx32++] = (uint32_t)result32;
            } else if (5 == first_difference) {
                output_ptr[o_idx32++] = (uint32_t)result32;
            }
        }
    } else {
        // We can pipeline the operations, so we do it four by four.
        for (size_t regist = 0; regist < num_registers; regist += 4) {
            size_t idx1 = (input_index_64 + regist);
            size_t idx2 = (input_index_64 + regist + 1);
            size_t idx3 = (input_index_64 + regist + 2);
            size_t idx4 = (input_index_64 + regist + 3);

            uint64_t original_implicant1 = implicants_ptr[idx1];
            uint64_t original_implicant2 = implicants_ptr[idx2];
            uint64_t original_implicant3 = implicants_ptr[idx3];
            uint64_t original_implicant4 = implicants_ptr[idx4];
            uint64_t primes1_0 = original_implicant1;
            uint64_t primes2_0 = original_implicant2;
            uint64_t primes3_0 = original_implicant3;
            uint64_t primes4_0 = original_implicant4;

            uint64_t aggregated1_1 = original_implicant1 & (original_implicant1 >> 1);
            uint64_t aggregated2_1 = original_implicant2 & (original_implicant2 >> 1);
            uint64_t aggregated3_1 = original_implicant3 & (original_implicant3 >> 1);
            uint64_t aggregated4_1 = original_implicant4 & (original_implicant4 >> 1);

            uint64_t aggregated1_2 = original_implicant1 & (original_implicant1 >> 2);
            uint64_t aggregated2_2 = original_implicant2 & (original_implicant2 >> 2);
            uint64_t aggregated3_2 = original_implicant3 & (original_implicant3 >> 2);
            uint64_t aggregated4_2 = original_implicant4 & (original_implicant4 >> 2);

            uint64_t aggregated1_4 = original_implicant1 & (original_implicant1 >> 4);
            uint64_t aggregated2_4 = original_implicant2 & (original_implicant2 >> 4);
            uint64_t aggregated3_4 = original_implicant3 & (original_implicant3 >> 4);
            uint64_t aggregated4_4 = original_implicant4 & (original_implicant4 >> 4);

            uint64_t aggregated1_8 = original_implicant1 & (original_implicant1 >> 8);
            uint64_t aggregated2_8 = original_implicant2 & (original_implicant2 >> 8);
            uint64_t aggregated3_8 = original_implicant3 & (original_implicant3 >> 8);
            uint64_t aggregated4_8 = original_implicant4 & (original_implicant4 >> 8);

            uint64_t aggregated1_16 = original_implicant1 & (original_implicant1 >> 16);
            uint64_t aggregated2_16 = original_implicant2 & (original_implicant2 >> 16);
            uint64_t aggregated3_16 = original_implicant3 & (original_implicant3 >> 16);
            uint64_t aggregated4_16 = original_implicant4 & (original_implicant4 >> 16);

            uint64_t aggregated1_32 = original_implicant1 & (original_implicant1 >> 32);
            uint64_t aggregated2_32 = original_implicant2 & (original_implicant2 >> 32);
            uint64_t aggregated3_32 = original_implicant3 & (original_implicant3 >> 32);
            uint64_t aggregated4_32 = original_implicant4 & (original_implicant4 >> 32);

            uint64_t result1_1 = _pext_u64(aggregated1_1, MASK1);
            uint64_t result2_1 = _pext_u64(aggregated2_1, MASK1);
            uint64_t result3_1 = _pext_u64(aggregated3_1, MASK1);
            uint64_t result4_1 = _pext_u64(aggregated4_1, MASK1);
            uint64_t ires1_1 = aggregated1_1 & MASK1;
            uint64_t ires2_1 = aggregated2_1 & MASK1;
            uint64_t ires3_1 = aggregated3_1 & MASK1;
            uint64_t ires4_1 = aggregated4_1 & MASK1;

            uint64_t result1_2 = _pext_u64(aggregated1_2, MASK2);
            uint64_t result2_2 = _pext_u64(aggregated2_2, MASK2);
            uint64_t result3_2 = _pext_u64(aggregated3_2, MASK2);
            uint64_t result4_2 = _pext_u64(aggregated4_2, MASK2);
            uint64_t ires1_2 = aggregated1_2 & MASK2;
            uint64_t ires2_2 = aggregated2_2 & MASK2;
            uint64_t ires3_2 = aggregated3_2 & MASK2;
            uint64_t ires4_2 = aggregated4_2 & MASK2;

            uint64_t result1_4 = _pext_u64(aggregated1_4, MASK4);
            uint64_t result2_4 = _pext_u64(aggregated2_4, MASK4);
            uint64_t result3_4 = _pext_u64(aggregated3_4, MASK4);
            uint64_t result4_4 = _pext_u64(aggregated4_4, MASK4);
            uint64_t ires1_4 = aggregated1_4 & MASK4;
            uint64_t ires2_4 = aggregated2_4 & MASK4;
            uint64_t ires3_4 = aggregated3_4 & MASK4;
            uint64_t ires4_4 = aggregated4_4 & MASK4;

            uint64_t result1_8 = _pext_u64(aggregated1_8, MASK8);
            uint64_t result2_8 = _pext_u64(aggregated2_8, MASK8);
            uint64_t result3_8 = _pext_u64(aggregated3_8, MASK8);
            uint64_t result4_8 = _pext_u64(aggregated4_8, MASK8);
            uint64_t ires1_8 = aggregated1_8 & MASK8;
            uint64_t ires2_8 = aggregated2_8 & MASK8;
            uint64_t ires3_8 = aggregated3_8 & MASK8;
            uint64_t ires4_8 = aggregated4_8 & MASK8;

            uint64_t result1_16 = _pext_u64(aggregated1_16, MASK16);
            uint64_t result2_16 = _pext_u64(aggregated2_16, MASK16);
            uint64_t result3_16 = _pext_u64(aggregated3_16, MASK16);
            uint64_t result4_16 = _pext_u64(aggregated4_16, MASK16);
            uint64_t ires1_16 = aggregated1_16 & MASK16;
            uint64_t ires2_16 = aggregated2_16 & MASK16;
            uint64_t ires3_16 = aggregated3_16 & MASK16;
            uint64_t ires4_16 = aggregated4_16 & MASK16;

            uint64_t result1_32 = _pext_u64(aggregated1_32, MASK32);
            uint64_t result2_32 = _pext_u64(aggregated2_32, MASK32);
            uint64_t result3_32 = _pext_u64(aggregated3_32, MASK32);
            uint64_t result4_32 = _pext_u64(aggregated4_32, MASK32);
            uint64_t ires1_32 = aggregated1_32 & MASK32;
            uint64_t ires2_32 = aggregated2_32 & MASK32;
            uint64_t ires3_32 = aggregated3_32 & MASK32;
            uint64_t ires4_32 = aggregated4_32 & MASK32;

            uint64_t primes1_1 = primes1_0 & ~(ires1_1 | (ires1_1 << 1));
            uint64_t primes2_1 = primes2_0 & ~(ires2_1 | (ires2_1 << 1));
            uint64_t primes3_1 = primes3_0 & ~(ires3_1 | (ires3_1 << 1));
            uint64_t primes4_1 = primes4_0 & ~(ires4_1 | (ires4_1 << 1));

            uint64_t primes1_2 = primes1_1 & ~(ires1_2 | (ires1_2 << 2));
            uint64_t primes2_2 = primes2_1 & ~(ires2_2 | (ires2_2 << 2));
            uint64_t primes3_2 = primes3_1 & ~(ires3_2 | (ires3_2 << 2));
            uint64_t primes4_2 = primes4_1 & ~(ires4_2 | (ires4_2 << 2));

            uint64_t primes1_4 = primes1_2 & ~(ires1_4 | (ires1_4 << 4));
            uint64_t primes2_4 = primes2_2 & ~(ires2_4 | (ires2_4 << 4));
            uint64_t primes3_4 = primes3_2 & ~(ires3_4 | (ires3_4 << 4));
            uint64_t primes4_4 = primes4_2 & ~(ires4_4 | (ires4_4 << 4));

            uint64_t primes1_8 = primes1_4 & ~(ires1_8 | (ires1_8 << 8));
            uint64_t primes2_8 = primes2_4 & ~(ires2_8 | (ires2_8 << 8));
            uint64_t primes3_8 = primes3_4 & ~(ires3_8 | (ires3_8 << 8));
            uint64_t primes4_8 = primes4_4 & ~(ires4_8 | (ires4_8 << 8));

            uint64_t primes1_16 = primes1_8 & ~(ires1_16 | (ires1_16 << 16));
            uint64_t primes2_16 = primes2_8 & ~(ires2_16 | (ires2_16 << 16));
            uint64_t primes3_16 = primes3_8 & ~(ires3_16 | (ires3_16 << 16));
            uint64_t primes4_16 = primes4_8 & ~(ires4_16 | (ires4_16 << 16));

            uint64_t primes1_32 = primes1_16 & ~(ires1_32 | (ires1_32 << 32));
            uint64_t primes2_32 = primes2_16 & ~(ires2_32 | (ires2_32 << 32));
            uint64_t primes3_32 = primes3_16 & ~(ires3_32 | (ires3_32 << 32));
            uint64_t primes4_32 = primes4_16 & ~(ires4_32 | (ires4_32 << 32));

            primes_ptr[idx1] = primes1_32;
            primes_ptr[idx2] = primes2_32;
            primes_ptr[idx3] = primes3_32;
            primes_ptr[idx4] = primes4_32;
            if (0 == first_difference) {
                output_ptr[o_idx1 + 0] = (uint32_t)result1_1;
                output_ptr[o_idx2 + 0] = (uint32_t)result1_2;
                output_ptr[o_idx4 + 0] = (uint32_t)result1_4;
                output_ptr[o_idx8 + 0] = (uint32_t)result1_8;
                output_ptr[o_idx16 + 0] = (uint32_t)result1_16;
                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;

                output_ptr[o_idx1 + 1] = (uint32_t)result2_1;
                output_ptr[o_idx2 + 1] = (uint32_t)result2_2;
                output_ptr[o_idx4 + 1] = (uint32_t)result2_4;
                output_ptr[o_idx8 + 1] = (uint32_t)result2_8;
                output_ptr[o_idx16 + 1] = (uint32_t)result2_16;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;

                output_ptr[o_idx1 + 2] = (uint32_t)result3_1;
                output_ptr[o_idx2 + 2] = (uint32_t)result3_2;
                output_ptr[o_idx4 + 2] = (uint32_t)result3_4;
                output_ptr[o_idx8 + 2] = (uint32_t)result3_8;
                output_ptr[o_idx16 + 2] = (uint32_t)result3_16;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;

                output_ptr[o_idx1 + 3] = (uint32_t)result4_1;
                output_ptr[o_idx2 + 3] = (uint32_t)result4_2;
                output_ptr[o_idx4 + 3] = (uint32_t)result4_4;
                output_ptr[o_idx8 + 3] = (uint32_t)result4_8;
                output_ptr[o_idx16 + 3] = (uint32_t)result4_16;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;

                o_idx1 += 4;
                o_idx2 += 4;
                o_idx4 += 4;
                o_idx8 += 4;
                o_idx16 += 4;
                o_idx32 += 4;
            } else if (1 == first_difference) {
                output_ptr[o_idx2 + 0] = (uint32_t)result1_2;
                output_ptr[o_idx4 + 0] = (uint32_t)result1_4;
                output_ptr[o_idx8 + 0] = (uint32_t)result1_8;
                output_ptr[o_idx16 + 0] = (uint32_t)result1_16;
                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;

                output_ptr[o_idx2 + 1] = (uint32_t)result2_2;
                output_ptr[o_idx4 + 1] = (uint32_t)result2_4;
                output_ptr[o_idx8 + 1] = (uint32_t)result2_8;
                output_ptr[o_idx16 + 1] = (uint32_t)result2_16;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;

                output_ptr[o_idx2 + 2] = (uint32_t)result3_2;
                output_ptr[o_idx4 + 2] = (uint32_t)result3_4;
                output_ptr[o_idx8 + 2] = (uint32_t)result3_8;
                output_ptr[o_idx16 + 2] = (uint32_t)result3_16;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;

                output_ptr[o_idx2 + 3] = (uint32_t)result4_2;
                output_ptr[o_idx4 + 3] = (uint32_t)result4_4;
                output_ptr[o_idx8 + 3] = (uint32_t)result4_8;
                output_ptr[o_idx16 + 3] = (uint32_t)result4_16;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;

                o_idx2 += 4;
                o_idx4 += 4;
                o_idx8 += 4;
                o_idx16 += 4;
                o_idx32 += 4;
            } else if (2 == first_difference) {
                output_ptr[o_idx4 + 0] = (uint32_t)result1_4;
                output_ptr[o_idx8 + 0] = (uint32_t)result1_8;
                output_ptr[o_idx16 + 0] = (uint32_t)result1_16;
                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;

                output_ptr[o_idx4 + 1] = (uint32_t)result2_4;
                output_ptr[o_idx8 + 1] = (uint32_t)result2_8;
                output_ptr[o_idx16 + 1] = (uint32_t)result2_16;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;

                output_ptr[o_idx4 + 2] = (uint32_t)result3_4;
                output_ptr[o_idx8 + 2] = (uint32_t)result3_8;
                output_ptr[o_idx16 + 2] = (uint32_t)result3_16;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;

                output_ptr[o_idx4 + 3] = (uint32_t)result4_4;
                output_ptr[o_idx8 + 3] = (uint32_t)result4_8;
                output_ptr[o_idx16 + 3] = (uint32_t)result4_16;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;

                o_idx4 += 4;
                o_idx8 += 4;
                o_idx16 += 4;
                o_idx32 += 4;
            } else if (3 == first_difference) {
                output_ptr[o_idx8 + 0] = (uint32_t)result1_8;
                output_ptr[o_idx16 + 0] = (uint32_t)result1_16;
                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;

                output_ptr[o_idx8 + 1] = (uint32_t)result2_8;
                output_ptr[o_idx16 + 1] = (uint32_t)result2_16;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;

                output_ptr[o_idx8 + 2] = (uint32_t)result3_8;
                output_ptr[o_idx16 + 2] = (uint32_t)result3_16;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;

                output_ptr[o_idx8 + 3] = (uint32_t)result4_8;
                output_ptr[o_idx16 + 3] = (uint32_t)result4_16;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;

                o_idx8 += 4;
                o_idx16 += 4;
                o_idx32 += 4;
            } else if (4 == first_difference) {
                output_ptr[o_idx16 + 0] = (uint32_t)result1_16;
                output_ptr[o_idx16 + 1] = (uint32_t)result2_16;
                output_ptr[o_idx16 + 2] = (uint32_t)result3_16;
                output_ptr[o_idx16 + 3] = (uint32_t)result4_16;

                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;

                o_idx16 += 4;
                o_idx32 += 4;
            } else if (5 == first_difference) {
                output_ptr[o_idx32 + 0] = (uint32_t)result1_32;
                output_ptr[o_idx32 + 1] = (uint32_t)result2_32;
                output_ptr[o_idx32 + 2] = (uint32_t)result3_32;
                output_ptr[o_idx32 + 3] = (uint32_t)result4_32;
                o_idx32 += 4;
            }
        }
    }
    // INTER REGISTER
    size_t o_idx64 = o_idx32 >> 1;
    int i = 6;
    for (; i < first_difference; i++) {
        int block_len_64 = 1 << (i - 6);  // Length of block in double words
        int num_blocks = 1 << (num_bits - i - 1);
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index_64 + 2 * block * block_len_64;
            size_t idx2 = input_index_64 + 2 * block * block_len_64 + block_len_64;

            for (int k = 0; k < block_len_64; k += 1) {
                uint64_t impl1 = implicants_ptr[idx1];
                uint64_t impl2 = implicants_ptr[idx2];
                uint64_t primes1 = primes_ptr[idx1];
                uint64_t primes2 = primes_ptr[idx2];
                uint64_t res = impl1 & impl2;
                uint64_t primes1_ = primes1 & ~res;
                uint64_t primes2_ = primes2 & ~res;

                primes_ptr[idx1++] = primes1_;
                primes_ptr[idx2++] = primes2_;
            }
        }
    }
    for (; i < num_bits; i++) {
        int block_len_64 = 1 << (i - 6);  // Length of block in double words
        int num_blocks = 1 << (num_bits - i - 1);
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index_64 + 2 * block * block_len_64;
            size_t idx2 = input_index_64 + 2 * block * block_len_64 + block_len_64;

            for (int k = 0; k < block_len_64; k += 1) {
                uint64_t impl1 = implicants_ptr[idx1];
                uint64_t impl2 = implicants_ptr[idx2];
                uint64_t primes1 = primes_ptr[idx1];
                uint64_t primes2 = primes_ptr[idx2];
                uint64_t res = impl1 & impl2;
                uint64_t primes1_ = primes1 & ~res;
                uint64_t primes2_ = primes2 & ~res;

                primes_ptr[idx1++] = primes1_;
                primes_ptr[idx2++] = primes2_;
                implicants_ptr[o_idx64++] = res;
            }
        }
    }
}
