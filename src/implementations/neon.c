#ifdef __aarch64__
#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"

#include "../vct_arm.h"
#include "../debug.h"

#include <arm_neon.h>
#include "bits.h"

static inline void merge_neon_single_register(
    int bit_difference,
    uint64x2_t impl1,
    uint64x2_t merged1,
    uint64_t  *result,
    uint64x2_t *merged_result
) {
    assert(0 <= bit_difference && bit_difference <= 6);

    int block_len = 1 << bit_difference;
    // if (block_len == 64) {
    //     // set upper half to zero, lower half to upper half of impl1
    //     uint64x2_t impl1_shuffled = vextq_u64(impl1, impl1, 1);
    //     uint64x2_t aggregated = vandq_u64(impl1, impl1_shuffled);
    //     // set upper and lower halves to lower half of aggregated
    //     uint64x2_t merged2 = vextq_u64(aggregated, aggregated, 1);
    //     // use lower half of aggregated
    //     *result = vuzpq_u64(aggregated, merged1).val[0];
    //     *merged_result = vorrq_u64(merged1, merged2);
    //     return;
    // }

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries

    // LOG_DEBUG("block_len: %d", block_len);

    // log_u64x2("impl1", &impl1);
    // log_u64x2_binary("impl1", &impl1);

    uint64x2_t impl2;
    if (block_len == 64) {
        // Shift across 64-bit boundaries (8 bytes = 64 bits)
        impl2 = vextq_u64(impl1, vdupq_n_u64(0), 1);
    } else {
        // // Shift within 64-bit lanes by a runtime-determined amount
        int64x2_t cnt = vdupq_n_s64(-block_len);     // negative => right shift
        impl2 = vshlq_u64(impl1, cnt);
    }   
    // log_u64x2("impl2", &impl2);
    // log_u64x2_binary("impl2", &impl2);
    uint64x2_t aggregated = vandq_u64(impl1, impl2);
    uint64x2_t initial_result = vdupq_n_u64(0); // prevent uninitialized warnings
    uint64x2_t shifted = vdupq_n_u64(0);

    // log_u64x2("aggregated1", &aggregated);
    // log_u64x2_binary("aggregated", &aggregated);

    if (block_len == 1) {
        aggregated = vandq_u64(aggregated, vdupq_n_u8(0b01010101));
        initial_result = aggregated;
        shifted = vshrq_n_u64(aggregated, 1);
    }
    // log_u64x2("aggregated", &aggregated);
    if (block_len <= 2) {
        aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vdupq_n_u8(0b00110011));
        if (block_len == 2) {
            initial_result = aggregated;
        }
        shifted = vshrq_n_u64(aggregated, 2);
    }
    if (block_len <= 4) {
        aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vdupq_n_u8(0b00001111));
        if (block_len == 4) {
            initial_result = aggregated;
        }
        shifted = vshrq_n_u64(aggregated, 4);
    }
    if (block_len <= 8) {
        aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vdupq_n_u16(0x00FF));
        if (block_len == 8) {
            initial_result = aggregated;
        }
        shifted = vshrq_n_u64(aggregated, 8);
    }
    if (block_len <= 16) {
        aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vdupq_n_u32(0x0000FFFF));
        if (block_len == 16) {
            initial_result = aggregated;
        }
        shifted = vshrq_n_u64(aggregated, 16);
    }
    if (block_len <= 32) {
        aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vdupq_n_u64(0x00000000FFFFFFFF));
        if (block_len == 32) {
            initial_result = aggregated;
        }
        uint32x4_t tmp = vreinterpretq_u32_u64(aggregated);
        uint32x4_t tmp_shifted = vextq_u32(tmp, tmp, 1);
        shifted = vreinterpretq_u64_u32(tmp_shifted);

    }
    // log_u64x2("aggregated", &aggregated);
    // log_u64x2("shifted", &shifted);
    // uint64x2_t shifted2 = vorrq_u64(aggregated, shifted);
    // uint64x2_t mask = vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0));
    // log_u64x2("aggregated", &aggregated);
    // log_u64x2("shifted2", &shifted2);
    // log_u64x2("mask", &mask);
    aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
    if (block_len == 64) {
        initial_result = aggregated;
    }

    

    // log_u64x2("aggregated_FINAL", &aggregated);
    // log_u64x2_binary("aggregated", &aggregated);
    // log_u64x2("initial_result", &initial_result);
    // Move 64-bit value aggregated[1] to result[1] so result is in the lower half
    // uint64x2_t result128 = vextq_u64(aggregated, aggregated, 1);

    *result = vgetq_lane_u64(aggregated, 0);
    // LOG_DEBUG("result: %lx", *result);
    // Perform bitwise OR between merged1 and initial_result
    uint64x2_t merged2 = vorrq_u64(merged1, initial_result);
    // log_u64x2("merged2", &merged2);
    // Shift initial_result left by block_len
    uint64x2_t merged3;
    if (block_len == 64) {
        // Shift left by 8 bytes (64 bits)
        merged3 = vextq_u64(vdupq_n_u64(0), initial_result, 1);
    } else {
            /// Shift left within 64-bit lanes
        int64x2_t shift_amount = vdupq_n_s64(block_len);
        merged3 = vshlq_u64(initial_result, shift_amount);
    }

    // log_u64x2("merged2", &merged2);
    // log_u64x2("merged3", &merged3);
    // Perform bitwise OR between merged2 and merged3
    *merged_result = vorrq_u64(merged2, merged3);
    // log_u64x2("merged_result", merged_result);

    return;
}

void merge_implicants_neon(
    bitmap implicants,
    bitmap merged,
    size_t input_index,
    size_t output_index,
    int num_bits,
    int first_difference
) 
{
    if (num_bits <= 6) {
        merge_implicants_bits(implicants, merged, input_index, output_index, num_bits, first_difference);
        return;
    }
    // LOG_DEBUG("Using merge_implicants_neon num_bits: %d, first_difference: %d", num_bits, first_difference);
    size_t o_idx = output_index;
    uint64_t *output_ptr = (uint64_t *) implicants.bits;

    for (int i = 0; i < num_bits; ++i) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 128)  {
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

                for (int k = 0; k < block_len; k += 128) {
                    uint64x2_t impl1 = vld1q_u64((uint64_t*)(implicants.bits + idx1/8));
                    uint64x2_t impl2 = vld1q_u64((uint64_t*)(implicants.bits + idx2/8));
                    uint64x2_t merged1 = vld1q_u64((uint64_t*)(merged.bits + idx1/8));
                    uint64x2_t merged2 = vld1q_u64((uint64_t*)(merged.bits + idx2/8));
                    uint64x2_t res = vandq_u64(impl1, impl2);
                    uint64x2_t merged1_ = vorrq_u64(merged1, res);
                    uint64x2_t merged2_ = vorrq_u64(merged2, res);
                    vst1q_u64((uint64_t*)(merged.bits + idx1/8), merged1_);
                    vst1q_u64((uint64_t*)(merged.bits + idx2/8), merged2_);
                    if (i >= first_difference) {
                        vst1q_u64((uint64_t*)(implicants.bits + o_idx/8), res);
                        o_idx += 128;
                    }
                    idx1 += 128;
                    idx2 += 128;
                }
            }
        }
        else {
            for (int block = 0; block < num_blocks; block += 64 / block_len) {
                size_t idx1 = input_index + 2 * block * block_len;

                uint64x2_t impl1 = vld1q_u64((uint64_t*)(implicants.bits + idx1/8));
                uint64x2_t merged1 = vld1q_u64((uint64_t*)(merged.bits + idx1/8));
                uint64_t impl_result;
                uint64x2_t merged_result;
                merge_neon_single_register(i, impl1, merged1, &impl_result, &merged_result);
                // LOG_DEBUG("impl_result: %lx", impl_result);
                // log_u64x2("merged_result", &merged_result);
                // print_bitmap_sparse("implicants_bitmap_before", &implicants);

                vst1q_u64((uint64_t*)(merged.bits + idx1/8), merged_result);
                // LOG_DEBUG("o_idx: %lu", o_idx);
                if (i >= first_difference) {
                    output_ptr[o_idx/64] = impl_result;
                    o_idx += 64;
                }
                // print_bitmap_sparse("implicants_bitmap_after", &implicants);
                // print_bitmap_bits(&implicants, 1600);
                idx1 += 128;
            }
        }
    }
}





prime_implicant_result prime_implicants_neon(
    int num_bits,
    int num_trues,
    int *trues
) {
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
    LOG_DEBUG("hihihihih");
    
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        // LOG_DEBUG("Remaining: %d, Num dashes: %d, Input Index: %d", remaining_bits, num_dashes, input_index);
        // print_bitmap_sparse("implicants", &implicants);
        // print_bitmap_sparse("merged", &merged);
        for (int j = 0; j < iterations; j++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, j);
            
            merge_implicants_neon(implicants, merged, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }
        // DEBUG DELETE BREAK!!!!

#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    LOG_DEBUG("After merge");
    LOG_DEBUG("num_implicants: %lu", num_implicants);
    // Step 2: Scan for unmerged implicants (process 64-bit lanes)
    for (size_t i = 0; i < num_implicants - (num_implicants % 128); i += 128) {
        // LOG_DEBUG("i: %lu", i);
        uint64x2_t impl = vld1q_u64((uint64_t*)(implicants.bits + i/8));
        uint64x2_t mer = vld1q_u64((uint64_t*)(merged.bits + i/8));
        // print impl
  
        // prime = impl & ~mer
        uint64x2_t prime = vbicq_u64(impl, mer);
        vst1q_u64((uint64_t*)(primes.bits + i/8), prime);

    }
    LOG_DEBUG("MIDDLE");
    for (size_t i = num_implicants - (num_implicants % 128); i < num_implicants - num_implicants % 64; i += 64) {
        // LOG_DEBUG("i: %lu", i);
        uint64_t implicant_true = ((uint64_t*)implicants.bits)[i / 64];
        uint64_t merged_true = ((uint64_t*)merged.bits)[i / 64];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t*)primes.bits)[i / 64] = prime_true;
    }
    LOG_DEBUG("LAST")
    for (size_t i = num_implicants - (num_implicants % 64); i < num_implicants; i++) {
        // LOG_DEBUG("i: %lu", i);
        if (BITMAP_CHECK(implicants, i) && !BITMAP_CHECK(merged, i)) {
            BITMAP_SET_TRUE(primes, i);
        }
    }
    // print primes
    // print_bitmap_sparse("implicants", &implicants);
    // print_bitmap_sparse("merged", &merged);
    // print_bitmap_sparse("primess", &primes);
    
    const uint64x2_t m0  = vdupq_n_u64(0x0101010101010101ULL);
    const uint64x2_t m1  = vdupq_n_u64(0x0101010101010101ULL);
    uint64x2_t x = vandq_u64(m0, m1);

    // 2) fold each 2-bit group into its low bit: OR with itself shifted right 1
    x = vorrq_u64(x, vshrq_n_u64(x, 1));
    //    now each bit-pair [2i+1,2i] has the selected bit in position 2i

    // 3) mask again to keep only bit-0 of each pair (so bits live at positions 0,2,4,…)
    x = vandq_u64(x, m1);
    log_u64x2("x", &x);
    // 4) reinterpret as 16×8-bit lanes so we can pull out one byte per selected bit
    uint8x16_t b = vreinterpretq_u8_u64(x);

    // 5) now we want to pack those 32 “bytes” (one per selected bit) down
    //    into the low half of each 64-bit lane.  We do this by treating each
    //    byte as a 0/1 and then doing a horizontal sum by pairs, then quads, etc:
    //
    //    a) sum adjacent bytes → 8×16-bit
    uint16x8_t s16 = vpaddlq_u8(b);
    //    b) sum adjacent 16-bit → 4×32-bit
    uint32x4_t s32 = vpaddlq_u16(s16);
    //    c) sum adjacent 32-bit → 2×64-bit
    uint64x2_t s64 = vpaddlq_u32(s32);
    log_u64x2("s64", &s64);

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