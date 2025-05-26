#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"
#include "../vtune.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../debug.h"

static void merge_implicants_bits6(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int first_difference) {
    uint64_t *implicant_ptr = (uint64_t *) implicants.bits;
    uint64_t *merged_ptr = (uint64_t *) merged.bits;
    uint32_t *output_ptr = (uint32_t *) implicants.bits;

    uint64_t impl = implicant_ptr[input_index / 64];
    uint64_t impl_merged = merged_ptr[input_index / 64];

    // block size 1
    uint64_t impl10 = impl & (impl >> 1) & 0b0101010101010101010101010101010101010101010101010101010101010101;
    uint64_t impl10s = impl10 >> 1;
    uint64_t impl11 = (impl10 | impl10s) & 0b0011001100110011001100110011001100110011001100110011001100110011;
    uint64_t impl11s = impl11 >> 2;
    uint64_t impl12 = (impl11 | impl11s) & 0x0F0F0F0F0F0F0F0F;
    uint64_t impl12s = impl12 >> 4;
    uint64_t impl13 = (impl12 | impl12s) & 0x00FF00FF00FF00FF;
    uint64_t impl13s = impl13 >> 8;
    uint64_t impl14 = (impl13 | impl13s) & 0x0000FFFF0000FFFF;
    uint64_t impl14s = impl14 >> 16;
    uint64_t impl15 = (impl14 | impl14s) & 0x00000000FFFFFFFF;
    uint32_t impl1res = (uint32_t) impl15;
    uint64_t merged1 = impl10 | (impl10 << 1);

    // block size 2
    uint64_t impl21 = impl & (impl >> 2) & 0b0011001100110011001100110011001100110011001100110011001100110011;
    uint64_t impl21s = impl21 >> 2;
    uint64_t impl22 = (impl21 | impl21s) & 0x0F0F0F0F0F0F0F0F;
    uint64_t impl22s = impl22 >> 4;
    uint64_t impl23 = (impl22 | impl22s) & 0x00FF00FF00FF00FF;
    uint64_t impl23s = impl23 >> 8;
    uint64_t impl24 = (impl23 | impl23s) & 0x0000FFFF0000FFFF;
    uint64_t impl24s = impl24 >> 16;
    uint64_t impl25 = (impl24 | impl24s) & 0x00000000FFFFFFFF;
    uint32_t impl2res = (uint32_t) impl25;
    uint64_t merged2 = impl21 | (impl21 << 2);

    // block size 4
    uint64_t impl32 = impl & (impl >> 4) & 0x0F0F0F0F0F0F0F0F;
    uint64_t impl32s = impl32 >> 4;
    uint64_t impl33 = (impl32 | impl32s) & 0x00FF00FF00FF00FF;
    uint64_t impl33s = impl33 >> 8;
    uint64_t impl34 = (impl33 | impl33s) & 0x0000FFFF0000FFFF;
    uint64_t impl34s = impl34 >> 16;
    uint64_t impl35 = (impl34 | impl34s) & 0x00000000FFFFFFFF;
    uint32_t impl3res = (uint32_t) impl35;
    uint64_t merged3 = impl32 | (impl32 << 4);

    // block size 8
    uint64_t impl43 = impl & (impl >> 8) & 0x00FF00FF00FF00FF;
    uint64_t impl43s = impl43 >> 8;
    uint64_t impl44 = (impl43 | impl43s) & 0x0000FFFF0000FFFF;
    uint64_t impl44s = impl44 >> 16;
    uint64_t impl45 = (impl44 | impl44s) & 0x00000000FFFFFFFF;
    uint32_t impl4res = (uint32_t) impl45;
    uint64_t merged4 = impl43 | (impl43 << 8);

    // block size 16
    uint64_t impl54 = impl & (impl >> 16) & 0x0000FFFF0000FFFF;
    uint64_t impl54s = impl54 >> 16;
    uint64_t impl55 = (impl54 | impl54s) & 0x00000000FFFFFFFF;
    uint32_t impl5res = (uint32_t) impl55;
    uint64_t merged5 = impl54 | (impl54 << 16);

    // block size 32
    uint64_t impl65 = impl & (impl >> 32) & 0x00000000FFFFFFFF;
    uint32_t impl6res = (uint32_t) impl65;
    uint64_t merged6 = impl65 | (impl65 << 32);

    uint64_t res_merged = impl_merged | merged1 | merged2 | merged3 | merged4 | merged5 | merged6;

    merged_ptr[input_index / 64] = res_merged;
    if (first_difference == 0) {
        output_ptr[(output_index / 32)] = impl1res;
        output_ptr[(output_index / 32)+1] = impl2res;
        output_ptr[(output_index / 32)+2] = impl3res;
        output_ptr[(output_index / 32)+3] = impl4res;
        output_ptr[(output_index / 32)+4] = impl5res;
        output_ptr[(output_index / 32)+5] = impl6res;
    } else if (first_difference == 1) {
        output_ptr[output_index / 32] = impl2res;
        output_ptr[(output_index / 32)+1] = impl3res;
        output_ptr[(output_index / 32)+2] = impl4res;
        output_ptr[(output_index / 32)+3] = impl5res;
        output_ptr[(output_index / 32)+4] = impl6res;
    } else if (first_difference == 2) {
        output_ptr[output_index / 32] = impl3res;
        output_ptr[(output_index / 32)+1] = impl4res;
        output_ptr[(output_index / 32)+2] = impl5res;
        output_ptr[(output_index / 32)+3] = impl6res;
    } else if (first_difference == 3) {
        output_ptr[output_index / 32] = impl4res;
        output_ptr[(output_index / 32)+1] = impl5res;
        output_ptr[(output_index / 32)+2] = impl6res;
    } else if (first_difference == 4) {
        output_ptr[output_index / 32] = impl5res;
        output_ptr[(output_index / 32)+1] = impl6res;
    } else if (first_difference == 5) {
        output_ptr[output_index / 32] = impl6res;
    }
}

static void merge_implicants_bits5(
    bitmap implicants,
    bitmap merged,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    uint32_t *implicant_ptr = (uint32_t *) implicants.bits;
    uint32_t *merged_ptr = (uint32_t *) merged.bits;
    uint16_t *output_ptr = (uint16_t *) implicants.bits; 

    uint32_t impl = implicant_ptr[input_index/32];
    uint32_t impl_merged = merged_ptr[input_index/32];

    // block size 1 (difference 2^0 = 1)
    uint32_t impl10 = impl & (impl >> 1) & 0x55555555;
    uint32_t impl10s = impl10 >> 1;
    uint32_t impl11 = (impl10 | impl10s) & 0x33333333;
    uint32_t impl11s = impl11 >> 2;
    uint32_t impl12 = (impl11 | impl11s) & 0x0F0F0F0F;
    uint32_t impl12s = impl12 >> 4;
    uint32_t impl13 = (impl12 | impl12s) & 0x00FF00FF;
    uint32_t impl13s = impl13 >> 8;
    uint32_t impl1res32 = (impl13 | impl13s) & 0x0000FFFF;
    uint32_t merged1 = impl10 | (impl10 << 1);

    // block size 2 (difference 2^1 = 2)
    uint32_t impl21 = impl & (impl >> 2) & 0x33333333;
    uint32_t impl21s = impl21 >> 2;
    uint32_t impl22 = (impl21 | impl21s) & 0x0F0F0F0F;
    uint32_t impl22s = impl22 >> 4;
    uint32_t impl23 = (impl22 | impl22s) & 0x00FF00FF;
    uint32_t impl23s = impl23 >> 8;
    uint32_t impl2res32 = (impl23 | impl23s) & 0x0000FFFF;
    uint32_t merged2 = impl21 | (impl21 << 2);

    // block size 4 (difference 2^2 = 4)
    uint32_t impl32 = impl & (impl >> 4) & 0x0F0F0F0F;
    uint32_t impl32s = impl32 >> 4;
    uint32_t impl33 = (impl32 | impl32s) & 0x00FF00FF;
    uint32_t impl33s = impl33 >> 8;
    uint32_t impl3res32 = (impl33 | impl33s) & 0x0000FFFF;
    uint32_t merged3 = impl32 | (impl32 << 4);

    // block size 8 (difference 2^3 = 8)
    uint32_t impl43 = impl & (impl >> 8) & 0x00FF00FF;
    uint32_t impl43s = impl43 >> 8;
    uint32_t impl4res32 = (impl43 | impl43s) & 0x0000FFFF;
    uint32_t merged4 = impl43 | (impl43 << 8);

    // block size 16 (difference 2^4 = 16)
    uint32_t impl5res32 = impl & (impl >> 16) & 0x0000FFFF;
    uint32_t merged5 = impl5res32 | (impl5res32 << 16);

    uint32_t res_merged = impl_merged | merged1 | merged2 | merged3 | merged4 | merged5;
    merged_ptr[input_index/32] = res_merged;

    if (first_difference == 0) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl1res32;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl2res32;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl3res32;
        output_ptr[(output_index/16) + 3] = (uint16_t)impl4res32;
        output_ptr[(output_index/16) + 4] = (uint16_t)impl5res32;
    } else if (first_difference == 1) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl2res32;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl3res32;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl4res32;
        output_ptr[(output_index/16) + 3] = (uint16_t)impl5res32;
    } else if (first_difference == 2) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl3res32;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl4res32;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl5res32;
    } else if (first_difference == 3) {
        output_ptr[(output_index/16)] = (uint16_t)impl4res32;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl5res32;
    } else if (first_difference == 4) {
        output_ptr[output_index/16] = (uint16_t)impl5res32;
    }
}

static void merge_implicants_bits1(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int first_difference) {
    bool impl1 = BITMAP_CHECK(implicants, input_index);
    bool impl2 = BITMAP_CHECK(implicants, input_index+1);
    bool merged1 = BITMAP_CHECK(merged, input_index);
    bool merged2 = BITMAP_CHECK(merged, input_index+1);

    bool res = impl1 && impl2;
    bool merged1_ = merged1 || res;
    bool merged2_ = merged2 || res;
    BITMAP_SET(merged, input_index, merged1_);
    BITMAP_SET(merged, input_index+1, merged2_);
    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res);
    }
}

static void merge_implicants_bits4(
    bitmap implicants,
    bitmap merged,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    uint16_t *implicant_ptr = (uint16_t *) implicants.bits;
    uint16_t *merged_ptr = (uint16_t *) merged.bits;
    uint8_t *output_ptr = (uint8_t *) implicants.bits;

    uint16_t impl = implicant_ptr[input_index/16];
    uint16_t impl_merged = merged_ptr[input_index/16];

    // block size 1 (difference 2^0 = 1)
    uint16_t impl10 = impl & (impl >> 1) & 0x5555;
    uint16_t impl10s = impl10 >> 1;
    uint16_t impl11 = (impl10 | impl10s) & 0x3333;
    uint16_t impl11s = impl11 >> 2;
    uint16_t impl12 = (impl11 | impl11s) & 0x0F0F;
    uint16_t impl12s = impl12 >> 4;
    uint16_t impl1res16 = (impl12 | impl12s) & 0x00FF; // Result fits in 8 bits
    uint16_t merged1 = impl10 | (impl10 << 1);

    // block size 2 (difference 2^1 = 2)
    uint16_t impl21 = impl & (impl >> 2) & 0x3333;
    uint16_t impl21s = impl21 >> 2;
    uint16_t impl22 = (impl21 | impl21s) & 0x0F0F;
    uint16_t impl22s = impl22 >> 4;
    uint16_t impl2res16 = (impl22 | impl22s) & 0x00FF;
    uint16_t merged2 = impl21 | (impl21 << 2);

    // block size 4 (difference 2^2 = 4)
    uint16_t impl32 = impl & (impl >> 4) & 0x0F0F;
    uint16_t impl32s = impl32 >> 4;
    uint16_t impl3res16 = (impl32 | impl32s) & 0x00FF;
    uint16_t merged3 = impl32 | (impl32 << 4);

    // block size 8 (difference 2^3 = 8)
    uint16_t impl4res16 = impl & (impl >> 8) & 0x00FF;
    uint16_t merged4 = impl4res16 | (impl4res16 << 8);

    // --- Combine merged markers (uint16_t) ---
    uint16_t res_merged = impl_merged | merged1 | merged2 | merged3 | merged4;
    merged_ptr[input_index/16] = res_merged;

    // --- Write output implicants (uint8_t) ---
    if (first_difference == 0) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl1res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl2res16;
        output_ptr[(output_index/8) + 2] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 3] = (uint8_t)impl4res16;
    } else if (first_difference == 1) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl2res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 2] = (uint8_t)impl4res16;
    } else if (first_difference == 2) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl4res16;
    } else if (first_difference == 3) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl4res16;
    }
}

static void merge_implicants_bits2(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int first_difference) {
    bool impl0 = BITMAP_CHECK(implicants, input_index);
    bool impl1 = BITMAP_CHECK(implicants, input_index+1);
    bool impl2 = BITMAP_CHECK(implicants, input_index+2);
    bool impl3 = BITMAP_CHECK(implicants, input_index+3);
    bool merged0 = BITMAP_CHECK(merged, input_index);
    bool merged1 = BITMAP_CHECK(merged, input_index+1);
    bool merged2 = BITMAP_CHECK(merged, input_index+2);
    bool merged3 = BITMAP_CHECK(merged, input_index+3);
    bool res0 = impl0 && impl1;
    bool res1 = impl2 && impl3;
    
    bool res2 = impl0 && impl2;
    bool res3 = impl1 && impl3;

    bool merged0_ = merged0 || res0 || res2;
    bool merged1_ = merged1 || res0 || res3;
    bool merged2_ = merged2 || res1 || res2;
    bool merged3_ = merged3 || res1 || res3;
    BITMAP_SET(merged, input_index, merged0_);
    BITMAP_SET(merged, input_index+1, merged1_);
    BITMAP_SET(merged, input_index+2, merged2_);
    BITMAP_SET(merged, input_index+3, merged3_);
    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res0);
        BITMAP_SET(implicants, output_index+1, res1);
        BITMAP_SET(implicants, output_index+2, res2);
        BITMAP_SET(implicants, output_index+3, res3);
    } else if (first_difference == 1) {
        BITMAP_SET(implicants, output_index, res2);
        BITMAP_SET(implicants, output_index+1, res3);
    }
}

static void merge_implicants_bits3(
    bitmap implicants,
    bitmap merged,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    bool impl0 = BITMAP_CHECK(implicants, input_index);
    bool impl1 = BITMAP_CHECK(implicants, input_index+1);
    bool impl2 = BITMAP_CHECK(implicants, input_index+2);
    bool impl3 = BITMAP_CHECK(implicants, input_index+3);
    bool impl4 = BITMAP_CHECK(implicants, input_index+4);
    bool impl5 = BITMAP_CHECK(implicants, input_index+5);
    bool impl6 = BITMAP_CHECK(implicants, input_index+6);
    bool impl7 = BITMAP_CHECK(implicants, input_index+7);
    bool merged0 = BITMAP_CHECK(merged, input_index);
    bool merged1 = BITMAP_CHECK(merged, input_index+1);
    bool merged2 = BITMAP_CHECK(merged, input_index+2);
    bool merged3 = BITMAP_CHECK(merged, input_index+3);
    bool merged4 = BITMAP_CHECK(merged, input_index+4);
    bool merged5 = BITMAP_CHECK(merged, input_index+5);
    bool merged6 = BITMAP_CHECK(merged, input_index+6);
    bool merged7 = BITMAP_CHECK(merged, input_index+7);
    bool res0 = impl0 && impl1;
    bool res1 = impl2 && impl3;
    bool res2 = impl4 && impl5;
    bool res3 = impl6 && impl7;
    bool res4 = impl0 && impl2;
    bool res5 = impl1 && impl3;
    bool res6 = impl4 && impl6;
    bool res7 = impl5 && impl7;
    bool res8 = impl0 && impl4;
    bool res9 = impl1 && impl5;
    bool res10 = impl2 && impl6;
    bool res11 = impl3 && impl7;
    bool merged0_ = merged0 || res0 || res4 || res8;
    bool merged1_ = merged1 || res0 || res5 || res9;
    bool merged2_ = merged2 || res1 || res4 || res10;
    bool merged3_ = merged3 || res1 || res5 || res11;
    bool merged4_ = merged4 || res2 || res6 || res8;
    bool merged5_ = merged5 || res2 || res7 || res9;
    bool merged6_ = merged6 || res3 || res6 || res10;
    bool merged7_ = merged7 || res3 || res7 || res11;
    BITMAP_SET(merged, input_index, merged0_);
    BITMAP_SET(merged, input_index+1, merged1_);
    BITMAP_SET(merged, input_index+2, merged2_);
    BITMAP_SET(merged, input_index+3, merged3_);
    BITMAP_SET(merged, input_index+4, merged4_);
    BITMAP_SET(merged, input_index+5, merged5_);
    BITMAP_SET(merged, input_index+6, merged6_);
    BITMAP_SET(merged, input_index+7, merged7_);

    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res0);
        BITMAP_SET(implicants, output_index+1, res1);
        BITMAP_SET(implicants, output_index+2, res2);
        BITMAP_SET(implicants, output_index+3, res3);
        BITMAP_SET(implicants, output_index+4, res4);
        BITMAP_SET(implicants, output_index+5, res5);
        BITMAP_SET(implicants, output_index+6, res6);
        BITMAP_SET(implicants, output_index+7, res7);
        BITMAP_SET(implicants, output_index+8, res8);
        BITMAP_SET(implicants, output_index+9, res9);
        BITMAP_SET(implicants, output_index+10, res10);
        BITMAP_SET(implicants, output_index+11, res11);
    } else if (first_difference == 1) {
        BITMAP_SET(implicants, output_index, res4);
        BITMAP_SET(implicants, output_index+1, res5);
        BITMAP_SET(implicants, output_index+2, res6);
        BITMAP_SET(implicants, output_index+3, res7);
        BITMAP_SET(implicants, output_index+4, res8);
        BITMAP_SET(implicants, output_index+5, res9);
        BITMAP_SET(implicants, output_index+6, res10);
        BITMAP_SET(implicants, output_index+7, res11);
    } else if (first_difference == 2) {
        BITMAP_SET(implicants, output_index, res8);
        BITMAP_SET(implicants, output_index+1, res9);
        BITMAP_SET(implicants, output_index+2, res10);
        BITMAP_SET(implicants, output_index+3, res11);
    }
}

void merge_implicants_bits(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits == 1) {
        merge_implicants_bits1(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 2) {
        merge_implicants_bits2(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 3) {
        merge_implicants_bits3(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 4) {
        merge_implicants_bits4(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 5) {
        merge_implicants_bits5(implicants, merged, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 6) {
        merge_implicants_bits6(implicants, merged, input_index, output_index, first_difference);
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
            
                    uint64_t merged2 = merged | initial_result | (initial_result << block_len);
    
                    merged_ptr[idx1 / 64] = merged2;
                    if (i >= first_difference) {
                        output_ptr[o_idx / 32] = (uint32_t) aggregated;
                        o_idx += 32;
                    }
        
                    idx1 += 64;
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
        ITT_START_TASK_SECTION(num_dashes);
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
        ITT_END_TASK();
#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    ITT_START_GATHER_TASK();
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
    ITT_END_TASK();
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