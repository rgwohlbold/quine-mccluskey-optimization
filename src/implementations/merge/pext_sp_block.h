#pragma once

#include <immintrin.h>

#include "../../bitmap.h"
#include "../../debug.h"
#include "bits_sp.h"
#ifndef LOG_BLOCK_SIZE_PEXT
#error "need to define LOG_BLOCK_SIZE_PEXT"
#endif

static inline void merge_pext_sp_block(bitmap implicants, bitmap primes, size_t input_index, size_t output_index,
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

    const size_t input_index_dw = input_index / 64;   // input_index is in bits, convert to 64-bit words
    const size_t output_index_w = output_index / 32;  // output_index is in bits, convert to 32-bit words
    size_t o_idx1 = output_index_w;
    size_t o_idx2 = output_index_w;
    // Layer size: 2^(num_bits - 1) /32 // Divided by 32 for index in 32-bit word writing
    size_t next_layer_size_w = (1 << (num_bits - 1 - 5));
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
    for (size_t regist = 0; regist < num_registers; regist++) {
        size_t idx1 = (input_index_dw + regist);

        uint64_t original_implicant = implicants_ptr[idx1];
        uint64_t primes = original_implicant;

        uint64_t aggregated1 = original_implicant & (original_implicant >> 1);
        uint64_t aggregated2 = original_implicant & (original_implicant >> 2);
        uint64_t aggregated4 = original_implicant & (original_implicant >> 4);
        uint64_t aggregated8 = original_implicant & (original_implicant >> 8);
        uint64_t aggregated16 = original_implicant & (original_implicant >> 16);
        uint64_t aggregated32 = original_implicant & (original_implicant >> 32);

#define MASK1 0b0101010101010101010101010101010101010101010101010101010101010101
#define MASK2 0b0011001100110011001100110011001100110011001100110011001100110011
#define MASK4 0x0F0F0F0F0F0F0F0F
#define MASK8 0x00FF00FF00FF00FF
#define MASK16 0x0000FFFF0000FFFF
#define MASK32 0x00000000FFFFFFFF

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

        uint64_t initial_result1_ = initial_result1 | (initial_result1 << 1);
        uint64_t initial_result2_ = initial_result2 | (initial_result2 << 2);
        uint64_t initial_result4_ = initial_result4 | (initial_result4 << 4);
        uint64_t initial_result8_ = initial_result8 | (initial_result8 << 8);
        uint64_t initial_result16_ = initial_result16 | (initial_result16 << 16);
        uint64_t initial_result32_ = initial_result32 | (initial_result32 << 32);

        uint64_t ir1 = initial_result1_ | initial_result2_;
        uint64_t ir2 = initial_result4_ | initial_result8_;
        uint64_t ir4 = initial_result16_ | initial_result32_;
        uint64_t initi = ir1 | ir2 | ir4;
        uint64_t primes32 = primes & ~initi;

        primes_ptr[idx1] = primes32;
        if (0 == first_difference) {
            output_ptr[o_idx1] = (uint32_t)result1;
            o_idx1 += 1;
            output_ptr[o_idx2] = (uint32_t)result2;
            o_idx2 += 1;
            output_ptr[o_idx4] = (uint32_t)result4;
            o_idx4 += 1;
            output_ptr[o_idx8] = (uint32_t)result8;
            o_idx8 += 1;
            output_ptr[o_idx16] = (uint32_t)result16;
            o_idx16 += 1;
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        } else if (1 == first_difference) {
            output_ptr[o_idx2] = (uint32_t)result2;
            o_idx2 += 1;
            output_ptr[o_idx4] = (uint32_t)result4;
            o_idx4 += 1;
            output_ptr[o_idx8] = (uint32_t)result8;
            o_idx8 += 1;
            output_ptr[o_idx16] = (uint32_t)result16;
            o_idx16 += 1;
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        } else if (2 == first_difference) {
            output_ptr[o_idx4] = (uint32_t)result4;
            o_idx4 += 1;
            output_ptr[o_idx8] = (uint32_t)result8;
            o_idx8 += 1;
            output_ptr[o_idx16] = (uint32_t)result16;
            o_idx16 += 1;
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        } else if (3 == first_difference) {
            output_ptr[o_idx8] = (uint32_t)result8;
            o_idx8 += 1;
            output_ptr[o_idx16] = (uint32_t)result16;
            o_idx16 += 1;
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        } else if (4 == first_difference) {
            output_ptr[o_idx16] = (uint32_t)result16;
            o_idx16 += 1;
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        } else if (5 == first_difference) {
            output_ptr[o_idx32] = (uint32_t)result32;
            o_idx32 += 1;
        }
    }

    // INTER REGISTER MERGES
    int i = 6;
    const size_t output_index_dw = output_index / 64;  // output_index is in bits, convert to 64-bit words

#if LOG_BLOCK_SIZE_PEXT >= 3
    {
        // Variables to store results
        uint64_t res_i_01, res_i_23, res_i_45, res_i_67, res_i_89, res_i_1011, res_i_1213, res_i_1415;
        uint64_t res_i1_02, res_i1_13, res_i1_46, res_i1_57, res_i1_810, res_i1_911, res_i1_1214, res_i1_1315;
        uint64_t res_i2_04, res_i2_15, res_i2_26, res_i2_37, res_i2_812, res_i2_913, res_i2_1014, res_i2_1115;
        uint64_t res_i3_08, res_i3_19, res_i3_210, res_i3_311, res_i3_412, res_i3_513, res_i3_614, res_i3_715;

        uint64_t p1_0, p1_1, p1_2, p1_3, p1_4, p1_5, p1_6, p1_7, p1_8, p1_9, p1_10, p1_11, p1_12, p1_13, p1_14, p1_15;
        uint64_t p2_0, p2_1, p2_2, p2_3, p2_4, p2_5, p2_6, p2_7, p2_8, p2_9, p2_10, p2_11, p2_12, p2_13, p2_14, p2_15;
        uint64_t p3_0, p3_1, p3_2, p3_3, p3_4, p3_5, p3_6, p3_7, p3_8, p3_9, p3_10, p3_11, p3_12, p3_13, p3_14, p3_15;
        uint64_t p4_0, p4_1, p4_2, p4_3, p4_4, p4_5, p4_6, p4_7, p4_8, p4_9, p4_10, p4_11, p4_12, p4_13, p4_14, p4_15;

        for (; i + 3 < first_difference; i += 4) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 8) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    {
                        uint64_t impl0 = implicants_ptr[idx1 + 0 * block_len];
                        uint64_t impl1 = implicants_ptr[idx1 + 1 * block_len];
                        uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                        uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];
                        uint64_t impl4 = implicants_ptr[idx1 + 4 * block_len];
                        uint64_t impl5 = implicants_ptr[idx1 + 5 * block_len];
                        uint64_t impl6 = implicants_ptr[idx1 + 6 * block_len];
                        uint64_t impl7 = implicants_ptr[idx1 + 7 * block_len];
                        uint64_t impl8 = implicants_ptr[idx1 + 8 * block_len];
                        uint64_t impl9 = implicants_ptr[idx1 + 9 * block_len];
                        uint64_t impl10 = implicants_ptr[idx1 + 10 * block_len];
                        uint64_t impl11 = implicants_ptr[idx1 + 11 * block_len];
                        uint64_t impl12 = implicants_ptr[idx1 + 12 * block_len];
                        uint64_t impl13 = implicants_ptr[idx1 + 13 * block_len];
                        uint64_t impl14 = implicants_ptr[idx1 + 14 * block_len];
                        uint64_t impl15 = implicants_ptr[idx1 + 15 * block_len];

                        uint64_t primes0 = primes_ptr[idx1 + 0 * block_len];
                        uint64_t primes1 = primes_ptr[idx1 + 1 * block_len];
                        uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                        uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];
                        uint64_t primes4 = primes_ptr[idx1 + 4 * block_len];
                        uint64_t primes5 = primes_ptr[idx1 + 5 * block_len];
                        uint64_t primes6 = primes_ptr[idx1 + 6 * block_len];
                        uint64_t primes7 = primes_ptr[idx1 + 7 * block_len];
                        uint64_t primes8 = primes_ptr[idx1 + 8 * block_len];
                        uint64_t primes9 = primes_ptr[idx1 + 9 * block_len];
                        uint64_t primes10 = primes_ptr[idx1 + 10 * block_len];
                        uint64_t primes11 = primes_ptr[idx1 + 11 * block_len];
                        uint64_t primes12 = primes_ptr[idx1 + 12 * block_len];
                        uint64_t primes13 = primes_ptr[idx1 + 13 * block_len];
                        uint64_t primes14 = primes_ptr[idx1 + 14 * block_len];
                        uint64_t primes15 = primes_ptr[idx1 + 15 * block_len];

                        // Compute results for all 4 levels
                        // Level i (distance 1)
                        res_i_01 = impl0 & impl1;
                        res_i_23 = impl2 & impl3;
                        res_i_45 = impl4 & impl5;
                        res_i_67 = impl6 & impl7;
                        res_i_89 = impl8 & impl9;
                        res_i_1011 = impl10 & impl11;
                        res_i_1213 = impl12 & impl13;
                        res_i_1415 = impl14 & impl15;

                        // Level i+1 (distance 2)
                        res_i1_02 = impl0 & impl2;
                        res_i1_13 = impl1 & impl3;
                        res_i1_46 = impl4 & impl6;
                        res_i1_57 = impl5 & impl7;
                        res_i1_810 = impl8 & impl10;
                        res_i1_911 = impl9 & impl11;
                        res_i1_1214 = impl12 & impl14;
                        res_i1_1315 = impl13 & impl15;

                        // Level i+2 (distance 4)
                        res_i2_04 = impl0 & impl4;
                        res_i2_15 = impl1 & impl5;
                        res_i2_26 = impl2 & impl6;
                        res_i2_37 = impl3 & impl7;
                        res_i2_812 = impl8 & impl12;
                        res_i2_913 = impl9 & impl13;
                        res_i2_1014 = impl10 & impl14;
                        res_i2_1115 = impl11 & impl15;

                        // Level i+3 (distance 8)
                        res_i3_08 = impl0 & impl8;
                        res_i3_19 = impl1 & impl9;
                        res_i3_210 = impl2 & impl10;
                        res_i3_311 = impl3 & impl11;
                        res_i3_412 = impl4 & impl12;
                        res_i3_513 = impl5 & impl13;
                        res_i3_614 = impl6 & impl14;
                        res_i3_715 = impl7 & impl15;

                        // Update primes - Stage 1
                        // _mm256_andnot_si256(A, B) is equivalent to (~A) & B
                        p1_0 = (~res_i_01) & primes0;
                        p1_1 = (~res_i_01) & primes1;
                        p1_2 = (~res_i_23) & primes2;
                        p1_3 = (~res_i_23) & primes3;
                        p1_4 = (~res_i_45) & primes4;
                        p1_5 = (~res_i_45) & primes5;
                        p1_6 = (~res_i_67) & primes6;
                        p1_7 = (~res_i_67) & primes7;
                        p1_8 = (~res_i_89) & primes8;
                        p1_9 = (~res_i_89) & primes9;
                        p1_10 = (~res_i_1011) & primes10;
                        p1_11 = (~res_i_1011) & primes11;
                        p1_12 = (~res_i_1213) & primes12;
                        p1_13 = (~res_i_1213) & primes13;
                        p1_14 = (~res_i_1415) & primes14;
                        p1_15 = (~res_i_1415) & primes15;

                        // Update primes - Stage 2
                        p2_0 = (~res_i1_02) & p1_0;
                        p2_1 = (~res_i1_13) & p1_1;
                        p2_2 = (~res_i1_02) & p1_2;
                        p2_3 = (~res_i1_13) & p1_3;
                        p2_4 = (~res_i1_46) & p1_4;
                        p2_5 = (~res_i1_57) & p1_5;
                        p2_6 = (~res_i1_46) & p1_6;
                        p2_7 = (~res_i1_57) & p1_7;
                        p2_8 = (~res_i1_810) & p1_8;
                        p2_9 = (~res_i1_911) & p1_9;
                        p2_10 = (~res_i1_810) & p1_10;
                        p2_11 = (~res_i1_911) & p1_11;
                        p2_12 = (~res_i1_1214) & p1_12;
                        p2_13 = (~res_i1_1315) & p1_13;
                        p2_14 = (~res_i1_1214) & p1_14;
                        p2_15 = (~res_i1_1315) & p1_15;

                        // Update primes - Stage 3
                        p3_0 = (~res_i2_04) & p2_0;
                        p3_1 = (~res_i2_15) & p2_1;
                        p3_2 = (~res_i2_26) & p2_2;
                        p3_3 = (~res_i2_37) & p2_3;
                        p3_4 = (~res_i2_04) & p2_4;
                        p3_5 = (~res_i2_15) & p2_5;
                        p3_6 = (~res_i2_26) & p2_6;
                        p3_7 = (~res_i2_37) & p2_7;
                        p3_8 = (~res_i2_812) & p2_8;
                        p3_9 = (~res_i2_913) & p2_9;
                        p3_10 = (~res_i2_1014) & p2_10;
                        p3_11 = (~res_i2_1115) & p2_11;
                        p3_12 = (~res_i2_812) & p2_12;
                        p3_13 = (~res_i2_913) & p2_13;
                        p3_14 = (~res_i2_1014) & p2_14;
                        p3_15 = (~res_i2_1115) & p2_15;

                        // Update primes - Stage 4 (final)
                        p4_0 = (~res_i3_08) & p3_0;
                        p4_1 = (~res_i3_19) & p3_1;
                        p4_2 = (~res_i3_210) & p3_2;
                        p4_3 = (~res_i3_311) & p3_3;
                        p4_4 = (~res_i3_412) & p3_4;
                        p4_5 = (~res_i3_513) & p3_5;
                        p4_6 = (~res_i3_614) & p3_6;
                        p4_7 = (~res_i3_715) & p3_7;
                        p4_8 = (~res_i3_08) & p3_8;
                        p4_9 = (~res_i3_19) & p3_9;
                        p4_10 = (~res_i3_210) & p3_10;
                        p4_11 = (~res_i3_311) & p3_11;
                        p4_12 = (~res_i3_412) & p3_12;
                        p4_13 = (~res_i3_513) & p3_13;
                        p4_14 = (~res_i3_614) & p3_14;
                        p4_15 = (~res_i3_715) & p3_15;

                        primes_ptr[idx1 + 0 * block_len] = p4_0;
                        primes_ptr[idx1 + 1 * block_len] = p4_1;
                        primes_ptr[idx1 + 2 * block_len] = p4_2;
                        primes_ptr[idx1 + 3 * block_len] = p4_3;
                        primes_ptr[idx1 + 4 * block_len] = p4_4;
                        primes_ptr[idx1 + 5 * block_len] = p4_5;
                        primes_ptr[idx1 + 6 * block_len] = p4_6;
                        primes_ptr[idx1 + 7 * block_len] = p4_7;
                        primes_ptr[idx1 + 8 * block_len] = p4_8;
                        primes_ptr[idx1 + 9 * block_len] = p4_9;
                        primes_ptr[idx1 + 10 * block_len] = p4_10;
                        primes_ptr[idx1 + 11 * block_len] = p4_11;
                        primes_ptr[idx1 + 12 * block_len] = p4_12;
                        primes_ptr[idx1 + 13 * block_len] = p4_13;
                        primes_ptr[idx1 + 14 * block_len] = p4_14;
                        primes_ptr[idx1 + 15 * block_len] = p4_15;
                    }

                    idx1 += 1;
                }
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_PEXT >= 2
    {
        for (; i + 2 < first_difference; i += 3) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 4) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl0 = implicants_ptr[idx1 + 0 * block_len];
                    uint64_t impl1 = implicants_ptr[idx1 + 1 * block_len];
                    uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                    uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];
                    uint64_t impl4 = implicants_ptr[idx1 + 4 * block_len];
                    uint64_t impl5 = implicants_ptr[idx1 + 5 * block_len];
                    uint64_t impl6 = implicants_ptr[idx1 + 6 * block_len];
                    uint64_t impl7 = implicants_ptr[idx1 + 7 * block_len];

                    uint64_t primes0 = primes_ptr[idx1 + 0 * block_len];
                    uint64_t primes1 = primes_ptr[idx1 + 1 * block_len];
                    uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                    uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];
                    uint64_t primes4 = primes_ptr[idx1 + 4 * block_len];
                    uint64_t primes5 = primes_ptr[idx1 + 5 * block_len];
                    uint64_t primes6 = primes_ptr[idx1 + 6 * block_len];
                    uint64_t primes7 = primes_ptr[idx1 + 7 * block_len];

                    uint64_t res01 = impl0 & impl1;
                    uint64_t res23 = impl2 & impl3;
                    uint64_t res45 = impl4 & impl5;
                    uint64_t res67 = impl6 & impl7;
                    uint64_t res02 = impl0 & impl2;
                    uint64_t res13 = impl1 & impl3;
                    uint64_t res46 = impl4 & impl6;
                    uint64_t res57 = impl5 & impl7;
                    uint64_t res04 = impl0 & impl4;
                    uint64_t res15 = impl1 & impl5;
                    uint64_t res26 = impl2 & impl6;
                    uint64_t res37 = impl3 & impl7;

                    // First stage of updating primes
                    uint64_t primes0_ = primes0 & ~res01;
                    uint64_t primes1_ = primes1 & ~res01;
                    uint64_t primes2_ = primes2 & ~res23;
                    uint64_t primes3_ = primes3 & ~res23;
                    uint64_t primes4_ = primes4 & ~res45;
                    uint64_t primes5_ = primes5 & ~res45;
                    uint64_t primes6_ = primes6 & ~res67;
                    uint64_t primes7_ = primes7 & ~res67;

                    // Second stage of updating primes
                    uint64_t primes0__ = primes0_ & ~res02;
                    uint64_t primes1__ = primes1_ & ~res13;
                    uint64_t primes2__ = primes2_ & ~res02;
                    uint64_t primes3__ = primes3_ & ~res13;
                    uint64_t primes4__ = primes4_ & ~res46;
                    uint64_t primes5__ = primes5_ & ~res57;
                    uint64_t primes6__ = primes6_ & ~res46;
                    uint64_t primes7__ = primes7_ & ~res57;

                    // Third stage of updating primes
                    uint64_t primes0___ = primes0__ & ~res04;
                    uint64_t primes1___ = primes1__ & ~res15;
                    uint64_t primes2___ = primes2__ & ~res26;
                    uint64_t primes3___ = primes3__ & ~res37;
                    uint64_t primes4___ = primes4__ & ~res04;
                    uint64_t primes5___ = primes5__ & ~res15;
                    uint64_t primes6___ = primes6__ & ~res26;
                    uint64_t primes7___ = primes7__ & ~res37;

                    primes_ptr[idx1 + 0 * block_len] = primes0___;
                    primes_ptr[idx1 + 1 * block_len] = primes1___;
                    primes_ptr[idx1 + 2 * block_len] = primes2___;
                    primes_ptr[idx1 + 3 * block_len] = primes3___;
                    primes_ptr[idx1 + 4 * block_len] = primes4___;
                    primes_ptr[idx1 + 5 * block_len] = primes5___;
                    primes_ptr[idx1 + 6 * block_len] = primes6___;
                    primes_ptr[idx1 + 7 * block_len] = primes7___;

                    idx1 += 1;
                }
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_PEXT >= 1
    {
        for (; i + 1 < first_difference; i += 2) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 2) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl0 = implicants_ptr[idx1];
                    uint64_t impl1 = implicants_ptr[idx1 + block_len];
                    uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                    uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];

                    uint64_t primes0 = primes_ptr[idx1];
                    uint64_t primes1 = primes_ptr[idx1 + block_len];
                    uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                    uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];

                    uint64_t res01 = impl0 & impl1;
                    uint64_t res23 = impl2 & impl3;
                    uint64_t res02 = impl0 & impl2;
                    uint64_t res13 = impl1 & impl3;

                    uint64_t primes0_ = primes0 & ~res01;
                    uint64_t primes1_ = primes1 & ~res01;
                    uint64_t primes2_ = primes2 & ~res23;
                    uint64_t primes3_ = primes3 & ~res23;

                    uint64_t primes0__ = primes0_ & ~res02;
                    uint64_t primes1__ = primes1_ & ~res13;
                    uint64_t primes2__ = primes2_ & ~res02;
                    uint64_t primes3__ = primes3_ & ~res13;

                    primes_ptr[idx1] = primes0__;
                    primes_ptr[idx1 + block_len] = primes1__;
                    primes_ptr[idx1 + 2 * block_len] = primes2__;
                    primes_ptr[idx1 + 3 * block_len] = primes3__;

                    idx1 += 1;
                }
            }
        }
    }

#endif
    {
        for (; i < first_difference; i++) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                size_t idx2 = (input_index_dw + (2 * block + 1) * block_len);
                for (int k = 0; k < block_len; k += 1) {
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
    }

    // After first_difference

#if LOG_BLOCK_SIZE_PEXT >= 3
    {
        // Variables to store results
        uint64_t res_i_01, res_i_23, res_i_45, res_i_67, res_i_89, res_i_1011, res_i_1213, res_i_1415;
        uint64_t res_i1_02, res_i1_13, res_i1_46, res_i1_57, res_i1_810, res_i1_911, res_i1_1214, res_i1_1315;
        uint64_t res_i2_04, res_i2_15, res_i2_26, res_i2_37, res_i2_812, res_i2_913, res_i2_1014, res_i2_1115;
        uint64_t res_i3_08, res_i3_19, res_i3_210, res_i3_311, res_i3_412, res_i3_513, res_i3_614, res_i3_715;

        uint64_t p1_0, p1_1, p1_2, p1_3, p1_4, p1_5, p1_6, p1_7, p1_8, p1_9, p1_10, p1_11, p1_12, p1_13, p1_14, p1_15;
        uint64_t p2_0, p2_1, p2_2, p2_3, p2_4, p2_5, p2_6, p2_7, p2_8, p2_9, p2_10, p2_11, p2_12, p2_13, p2_14, p2_15;
        uint64_t p3_0, p3_1, p3_2, p3_3, p3_4, p3_5, p3_6, p3_7, p3_8, p3_9, p3_10, p3_11, p3_12, p3_13, p3_14, p3_15;
        uint64_t p4_0, p4_1, p4_2, p4_3, p4_4, p4_5, p4_6, p4_7, p4_8, p4_9, p4_10, p4_11, p4_12, p4_13, p4_14, p4_15;

        for (; i + 3 < num_bits; i += 4) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 8) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl0 = implicants_ptr[idx1 + 0 * block_len];
                    uint64_t impl1 = implicants_ptr[idx1 + 1 * block_len];
                    uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                    uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];
                    uint64_t impl4 = implicants_ptr[idx1 + 4 * block_len];
                    uint64_t impl5 = implicants_ptr[idx1 + 5 * block_len];
                    uint64_t impl6 = implicants_ptr[idx1 + 6 * block_len];
                    uint64_t impl7 = implicants_ptr[idx1 + 7 * block_len];
                    uint64_t impl8 = implicants_ptr[idx1 + 8 * block_len];
                    uint64_t impl9 = implicants_ptr[idx1 + 9 * block_len];
                    uint64_t impl10 = implicants_ptr[idx1 + 10 * block_len];
                    uint64_t impl11 = implicants_ptr[idx1 + 11 * block_len];
                    uint64_t impl12 = implicants_ptr[idx1 + 12 * block_len];
                    uint64_t impl13 = implicants_ptr[idx1 + 13 * block_len];
                    uint64_t impl14 = implicants_ptr[idx1 + 14 * block_len];
                    uint64_t impl15 = implicants_ptr[idx1 + 15 * block_len];

                    uint64_t primes0 = primes_ptr[idx1 + 0 * block_len];
                    uint64_t primes1 = primes_ptr[idx1 + 1 * block_len];
                    uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                    uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];
                    uint64_t primes4 = primes_ptr[idx1 + 4 * block_len];
                    uint64_t primes5 = primes_ptr[idx1 + 5 * block_len];
                    uint64_t primes6 = primes_ptr[idx1 + 6 * block_len];
                    uint64_t primes7 = primes_ptr[idx1 + 7 * block_len];
                    uint64_t primes8 = primes_ptr[idx1 + 8 * block_len];
                    uint64_t primes9 = primes_ptr[idx1 + 9 * block_len];
                    uint64_t primes10 = primes_ptr[idx1 + 10 * block_len];
                    uint64_t primes11 = primes_ptr[idx1 + 11 * block_len];
                    uint64_t primes12 = primes_ptr[idx1 + 12 * block_len];
                    uint64_t primes13 = primes_ptr[idx1 + 13 * block_len];
                    uint64_t primes14 = primes_ptr[idx1 + 14 * block_len];
                    uint64_t primes15 = primes_ptr[idx1 + 15 * block_len];

                    // Compute results for all 4 levels
                    // Level i (distance 1)
                    res_i_01 = impl0 & impl1;
                    res_i_23 = impl2 & impl3;
                    res_i_45 = impl4 & impl5;
                    res_i_67 = impl6 & impl7;
                    res_i_89 = impl8 & impl9;
                    res_i_1011 = impl10 & impl11;
                    res_i_1213 = impl12 & impl13;
                    res_i_1415 = impl14 & impl15;

                    // Level i+1 (distance 2)
                    res_i1_02 = impl0 & impl2;
                    res_i1_13 = impl1 & impl3;
                    res_i1_46 = impl4 & impl6;
                    res_i1_57 = impl5 & impl7;
                    res_i1_810 = impl8 & impl10;
                    res_i1_911 = impl9 & impl11;
                    res_i1_1214 = impl12 & impl14;
                    res_i1_1315 = impl13 & impl15;

                    // Level i+2 (distance 4)
                    res_i2_04 = impl0 & impl4;
                    res_i2_15 = impl1 & impl5;
                    res_i2_26 = impl2 & impl6;
                    res_i2_37 = impl3 & impl7;
                    res_i2_812 = impl8 & impl12;
                    res_i2_913 = impl9 & impl13;
                    res_i2_1014 = impl10 & impl14;
                    res_i2_1115 = impl11 & impl15;

                    // Level i+3 (distance 8)
                    res_i3_08 = impl0 & impl8;
                    res_i3_19 = impl1 & impl9;
                    res_i3_210 = impl2 & impl10;
                    res_i3_311 = impl3 & impl11;
                    res_i3_412 = impl4 & impl12;
                    res_i3_513 = impl5 & impl13;
                    res_i3_614 = impl6 & impl14;
                    res_i3_715 = impl7 & impl15;

                    // Update primes - Stage 1
                    // _mm256_andnot_si256(A, B) is equivalent to (~A) & B
                    p1_0 = (~res_i_01) & primes0;
                    p1_1 = (~res_i_01) & primes1;
                    p1_2 = (~res_i_23) & primes2;
                    p1_3 = (~res_i_23) & primes3;
                    p1_4 = (~res_i_45) & primes4;
                    p1_5 = (~res_i_45) & primes5;
                    p1_6 = (~res_i_67) & primes6;
                    p1_7 = (~res_i_67) & primes7;
                    p1_8 = (~res_i_89) & primes8;
                    p1_9 = (~res_i_89) & primes9;
                    p1_10 = (~res_i_1011) & primes10;
                    p1_11 = (~res_i_1011) & primes11;
                    p1_12 = (~res_i_1213) & primes12;
                    p1_13 = (~res_i_1213) & primes13;
                    p1_14 = (~res_i_1415) & primes14;
                    p1_15 = (~res_i_1415) & primes15;

                    // Update primes - Stage 2
                    p2_0 = (~res_i1_02) & p1_0;
                    p2_1 = (~res_i1_13) & p1_1;
                    p2_2 = (~res_i1_02) & p1_2;
                    p2_3 = (~res_i1_13) & p1_3;
                    p2_4 = (~res_i1_46) & p1_4;
                    p2_5 = (~res_i1_57) & p1_5;
                    p2_6 = (~res_i1_46) & p1_6;
                    p2_7 = (~res_i1_57) & p1_7;
                    p2_8 = (~res_i1_810) & p1_8;
                    p2_9 = (~res_i1_911) & p1_9;
                    p2_10 = (~res_i1_810) & p1_10;
                    p2_11 = (~res_i1_911) & p1_11;
                    p2_12 = (~res_i1_1214) & p1_12;
                    p2_13 = (~res_i1_1315) & p1_13;
                    p2_14 = (~res_i1_1214) & p1_14;
                    p2_15 = (~res_i1_1315) & p1_15;

                    // Update primes - Stage 3
                    p3_0 = (~res_i2_04) & p2_0;
                    p3_1 = (~res_i2_15) & p2_1;
                    p3_2 = (~res_i2_26) & p2_2;
                    p3_3 = (~res_i2_37) & p2_3;
                    p3_4 = (~res_i2_04) & p2_4;
                    p3_5 = (~res_i2_15) & p2_5;
                    p3_6 = (~res_i2_26) & p2_6;
                    p3_7 = (~res_i2_37) & p2_7;
                    p3_8 = (~res_i2_812) & p2_8;
                    p3_9 = (~res_i2_913) & p2_9;
                    p3_10 = (~res_i2_1014) & p2_10;
                    p3_11 = (~res_i2_1115) & p2_11;
                    p3_12 = (~res_i2_812) & p2_12;
                    p3_13 = (~res_i2_913) & p2_13;
                    p3_14 = (~res_i2_1014) & p2_14;
                    p3_15 = (~res_i2_1115) & p2_15;

                    // Update primes - Stage 4 (final)
                    p4_0 = (~res_i3_08) & p3_0;
                    p4_1 = (~res_i3_19) & p3_1;
                    p4_2 = (~res_i3_210) & p3_2;
                    p4_3 = (~res_i3_311) & p3_3;
                    p4_4 = (~res_i3_412) & p3_4;
                    p4_5 = (~res_i3_513) & p3_5;
                    p4_6 = (~res_i3_614) & p3_6;
                    p4_7 = (~res_i3_715) & p3_7;
                    p4_8 = (~res_i3_08) & p3_8;
                    p4_9 = (~res_i3_19) & p3_9;
                    p4_10 = (~res_i3_210) & p3_10;
                    p4_11 = (~res_i3_311) & p3_11;
                    p4_12 = (~res_i3_412) & p3_12;
                    p4_13 = (~res_i3_513) & p3_13;
                    p4_14 = (~res_i3_614) & p3_14;
                    p4_15 = (~res_i3_715) & p3_15;

                    primes_ptr[idx1 + 0 * block_len] = p4_0;
                    primes_ptr[idx1 + 1 * block_len] = p4_1;
                    primes_ptr[idx1 + 2 * block_len] = p4_2;
                    primes_ptr[idx1 + 3 * block_len] = p4_3;
                    primes_ptr[idx1 + 4 * block_len] = p4_4;
                    primes_ptr[idx1 + 5 * block_len] = p4_5;
                    primes_ptr[idx1 + 6 * block_len] = p4_6;
                    primes_ptr[idx1 + 7 * block_len] = p4_7;
                    primes_ptr[idx1 + 8 * block_len] = p4_8;
                    primes_ptr[idx1 + 9 * block_len] = p4_9;
                    primes_ptr[idx1 + 10 * block_len] = p4_10;
                    primes_ptr[idx1 + 11 * block_len] = p4_11;
                    primes_ptr[idx1 + 12 * block_len] = p4_12;
                    primes_ptr[idx1 + 13 * block_len] = p4_13;
                    primes_ptr[idx1 + 14 * block_len] = p4_14;
                    primes_ptr[idx1 + 15 * block_len] = p4_15;

                    {
                        size_t o =
                            output_index_dw + ((i - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res_i_01;
                        implicants_ptr[o + 1 * block_len] = res_i_23;
                        implicants_ptr[o + 2 * block_len] = res_i_45;
                        implicants_ptr[o + 3 * block_len] = res_i_67;
                        implicants_ptr[o + 4 * block_len] = res_i_89;
                        implicants_ptr[o + 5 * block_len] = res_i_1011;
                        implicants_ptr[o + 6 * block_len] = res_i_1213;
                        implicants_ptr[o + 7 * block_len] = res_i_1415;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 1 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res_i1_02;
                        implicants_ptr[o + 1 * block_len] = res_i1_13;
                        implicants_ptr[o + 2 * block_len] = res_i1_46;
                        implicants_ptr[o + 3 * block_len] = res_i1_57;
                        implicants_ptr[o + 4 * block_len] = res_i1_810;
                        implicants_ptr[o + 5 * block_len] = res_i1_911;
                        implicants_ptr[o + 6 * block_len] = res_i1_1214;
                        implicants_ptr[o + 7 * block_len] = res_i1_1315;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 2 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res_i2_04;
                        implicants_ptr[o + 1 * block_len] = res_i2_15;
                        implicants_ptr[o + 2 * block_len] = res_i2_26;
                        implicants_ptr[o + 3 * block_len] = res_i2_37;
                        implicants_ptr[o + 4 * block_len] = res_i2_812;
                        implicants_ptr[o + 5 * block_len] = res_i2_913;
                        implicants_ptr[o + 6 * block_len] = res_i2_1014;
                        implicants_ptr[o + 7 * block_len] = res_i2_1115;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 3 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res_i3_08;
                        implicants_ptr[o + 1 * block_len] = res_i3_19;
                        implicants_ptr[o + 2 * block_len] = res_i3_210;
                        implicants_ptr[o + 3 * block_len] = res_i3_311;
                        implicants_ptr[o + 4 * block_len] = res_i3_412;
                        implicants_ptr[o + 5 * block_len] = res_i3_513;
                        implicants_ptr[o + 6 * block_len] = res_i3_614;
                        implicants_ptr[o + 7 * block_len] = res_i3_715;
                    }
                    idx1 += 1;
                }
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_PEXT >= 2
    {
        for (; i + 2 < num_bits; i += 3) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 4) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl0 = implicants_ptr[idx1 + 0 * block_len];
                    uint64_t impl1 = implicants_ptr[idx1 + 1 * block_len];
                    uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                    uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];
                    uint64_t impl4 = implicants_ptr[idx1 + 4 * block_len];
                    uint64_t impl5 = implicants_ptr[idx1 + 5 * block_len];
                    uint64_t impl6 = implicants_ptr[idx1 + 6 * block_len];
                    uint64_t impl7 = implicants_ptr[idx1 + 7 * block_len];

                    uint64_t primes0 = primes_ptr[idx1 + 0 * block_len];
                    uint64_t primes1 = primes_ptr[idx1 + 1 * block_len];
                    uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                    uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];
                    uint64_t primes4 = primes_ptr[idx1 + 4 * block_len];
                    uint64_t primes5 = primes_ptr[idx1 + 5 * block_len];
                    uint64_t primes6 = primes_ptr[idx1 + 6 * block_len];
                    uint64_t primes7 = primes_ptr[idx1 + 7 * block_len];

                    uint64_t res01 = impl0 & impl1;
                    uint64_t res23 = impl2 & impl3;
                    uint64_t res45 = impl4 & impl5;
                    uint64_t res67 = impl6 & impl7;
                    uint64_t res02 = impl0 & impl2;
                    uint64_t res13 = impl1 & impl3;
                    uint64_t res46 = impl4 & impl6;
                    uint64_t res57 = impl5 & impl7;
                    uint64_t res04 = impl0 & impl4;
                    uint64_t res15 = impl1 & impl5;
                    uint64_t res26 = impl2 & impl6;
                    uint64_t res37 = impl3 & impl7;

                    // First stage of updating primes
                    uint64_t primes0_ = primes0 & ~res01;
                    uint64_t primes1_ = primes1 & ~res01;
                    uint64_t primes2_ = primes2 & ~res23;
                    uint64_t primes3_ = primes3 & ~res23;
                    uint64_t primes4_ = primes4 & ~res45;
                    uint64_t primes5_ = primes5 & ~res45;
                    uint64_t primes6_ = primes6 & ~res67;
                    uint64_t primes7_ = primes7 & ~res67;

                    // Second stage of updating primes
                    uint64_t primes0__ = primes0_ & ~res02;
                    uint64_t primes1__ = primes1_ & ~res13;
                    uint64_t primes2__ = primes2_ & ~res02;
                    uint64_t primes3__ = primes3_ & ~res13;
                    uint64_t primes4__ = primes4_ & ~res46;
                    uint64_t primes5__ = primes5_ & ~res57;
                    uint64_t primes6__ = primes6_ & ~res46;
                    uint64_t primes7__ = primes7_ & ~res57;

                    // Third stage of updating primes
                    uint64_t primes0___ = primes0__ & ~res04;
                    uint64_t primes1___ = primes1__ & ~res15;
                    uint64_t primes2___ = primes2__ & ~res26;
                    uint64_t primes3___ = primes3__ & ~res37;
                    uint64_t primes4___ = primes4__ & ~res04;
                    uint64_t primes5___ = primes5__ & ~res15;
                    uint64_t primes6___ = primes6__ & ~res26;
                    uint64_t primes7___ = primes7__ & ~res37;

                    primes_ptr[idx1 + 0 * block_len] = primes0___;
                    primes_ptr[idx1 + 1 * block_len] = primes1___;
                    primes_ptr[idx1 + 2 * block_len] = primes2___;
                    primes_ptr[idx1 + 3 * block_len] = primes3___;
                    primes_ptr[idx1 + 4 * block_len] = primes4___;
                    primes_ptr[idx1 + 5 * block_len] = primes5___;
                    primes_ptr[idx1 + 6 * block_len] = primes6___;
                    primes_ptr[idx1 + 7 * block_len] = primes7___;

                    {
                        size_t o =
                            output_index_dw + ((i - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res01;
                        implicants_ptr[o + 1 * block_len] = res23;
                        implicants_ptr[o + 2 * block_len] = res45;
                        implicants_ptr[o + 3 * block_len] = res67;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 1 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res02;
                        implicants_ptr[o + 1 * block_len] = res13;
                        implicants_ptr[o + 2 * block_len] = res46;
                        implicants_ptr[o + 3 * block_len] = res57;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 2 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o + 0 * block_len] = res04;
                        implicants_ptr[o + 1 * block_len] = res15;
                        implicants_ptr[o + 2 * block_len] = res26;
                        implicants_ptr[o + 3 * block_len] = res37;
                    }
                    idx1 += 1;
                }
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_PEXT >= 1
    {
        for (; i + 1 < num_bits; i += 2) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block += 2) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl0 = implicants_ptr[idx1];
                    uint64_t impl1 = implicants_ptr[idx1 + block_len];
                    uint64_t impl2 = implicants_ptr[idx1 + 2 * block_len];
                    uint64_t impl3 = implicants_ptr[idx1 + 3 * block_len];

                    uint64_t primes0 = primes_ptr[idx1];
                    uint64_t primes1 = primes_ptr[idx1 + block_len];
                    uint64_t primes2 = primes_ptr[idx1 + 2 * block_len];
                    uint64_t primes3 = primes_ptr[idx1 + 3 * block_len];

                    uint64_t res01 = impl0 & impl1;
                    uint64_t res23 = impl2 & impl3;
                    uint64_t res02 = impl0 & impl2;
                    uint64_t res13 = impl1 & impl3;

                    uint64_t primes0_ = primes0 & ~res01;
                    uint64_t primes1_ = primes1 & ~res01;
                    uint64_t primes2_ = primes2 & ~res23;
                    uint64_t primes3_ = primes3 & ~res23;

                    uint64_t primes0__ = primes0_ & ~res02;
                    uint64_t primes1__ = primes1_ & ~res13;
                    uint64_t primes2__ = primes2_ & ~res02;
                    uint64_t primes3__ = primes3_ & ~res13;

                    primes_ptr[idx1] = primes0__;
                    primes_ptr[idx1 + block_len] = primes1__;
                    primes_ptr[idx1 + 2 * block_len] = primes2__;
                    primes_ptr[idx1 + 3 * block_len] = primes3__;

                    {
                        size_t o =
                            output_index_dw + ((i - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o] = res01;
                        implicants_ptr[o + block_len] = res23;
                    }
                    {
                        size_t o =
                            output_index_dw + ((i + 1 - first_difference) << (num_bits - 7)) + (block * block_len + k);
                        implicants_ptr[o] = res02;
                        implicants_ptr[o + block_len] = res13;
                    }
                    idx1 += 1;
                }
            }
        }
    }

#endif
    {
        for (; i < num_bits; i++) {
            // Length of block in double words
            int block_len = (1 << (i - 6));
            int num_blocks = 1 << (num_bits - i - 1);
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = (input_index_dw + 2 * block * block_len);
                size_t idx2 = (input_index_dw + (2 * block + 1) * block_len);
                for (int k = 0; k < block_len; k += 1) {
                    uint64_t impl1 = implicants_ptr[idx1];
                    uint64_t impl2 = implicants_ptr[idx2];

                    uint64_t primes1 = primes_ptr[idx1];
                    uint64_t primes2 = primes_ptr[idx2];

                    uint64_t res = impl1 & impl2;

                    uint64_t primes1_ = primes1 & ~res;
                    uint64_t primes2_ = primes2 & ~res;

                    primes_ptr[idx1] = primes1_;
                    primes_ptr[idx2] = primes2_;
                    size_t o = output_index_dw + ((i - first_difference) << (num_bits - 7)) + (block * block_len + k);
                    implicants_ptr[o] = res;
                    idx1 += 1;
                    idx2 += 1;
                }
            }
        }
    }
}
