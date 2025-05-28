#ifdef __aarch64__
#include "bits_sp.h"

static inline void merge_neon_sp_single_register(
    int bit_difference,
    uint64x2_t impl1,
    uint64x2_t primes1,
    uint64_t  *result,
    uint64x2_t *primes_result
) {
    assert(0 <= bit_difference && bit_difference <= 6);

    int block_len = 1 << bit_difference;

    uint64x2_t impl2;
    if (block_len == 64) {
        // Shift across 64-bit boundaries (8 bytes = 64 bits)
        impl2 = vextq_u64(impl1, vdupq_n_u64(0), 1);
    } else {
        // // Shift within 64-bit lanes by a runtime-determined amount
        int64x2_t cnt = vdupq_n_s64(-block_len);     // negative => right shift
        impl2 = vshlq_u64(impl1, cnt);
    }

    uint64x2_t aggregated = vandq_u64(impl1, impl2);
    uint64x2_t initial_result = vdupq_n_u64(0); // prevent uninitialized warnings
    uint64x2_t shifted = vdupq_n_u64(0);


    if (block_len == 1) {
        aggregated = vandq_u64(aggregated, vdupq_n_u8(0b01010101));
        initial_result = aggregated;
        shifted = vshrq_n_u64(aggregated, 1);
    }
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

    aggregated = vandq_u64(vorrq_u64(aggregated, shifted), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
    if (block_len == 64) {
        initial_result = aggregated;
    }

    // Move 64-bit value aggregated[1] to result[1] so result is in the lower half
    *result = vgetq_lane_u64(aggregated, 0);
    // Shift initial_result left by block_len
    uint64x2_t merged2;
    if (block_len == 64) {
        // Shift left by 8 bytes (64 bits)
        merged2 = vextq_u64(vdupq_n_u64(0), initial_result, 1);
    } else {
            /// Shift left within 64-bit lanes
        int64x2_t shift_amount = vdupq_n_s64(block_len);
        merged2 = vshlq_u64(initial_result, shift_amount);
    }
    merged2 = vorrq_u64(merged2, initial_result);
    *primes_result = vbicq_u64(primes1, merged2);

    return;
}

static inline void merge_neon_sp(
    bitmap implicants,
    bitmap primes,
    size_t input_index,
    size_t output_index,
    int num_bits,
    int first_difference
)
{
    if (num_bits <= 6) {
        merge_bits_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }
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
                    uint64x2_t primes1 = vld1q_u64((uint64_t*)(primes.bits + idx1/8));
                    uint64x2_t primes2 = vld1q_u64((uint64_t*)(primes.bits + idx2/8));
                    uint64x2_t res = vandq_u64(impl1, impl2);
                    uint64x2_t primes1_ = vbicq_u64(primes1, res);
                    uint64x2_t primes2_ = vbicq_u64(primes2, res);
                    vst1q_u64((uint64_t*)(primes.bits + idx1/8), primes1_);
                    vst1q_u64((uint64_t*)(primes.bits + idx2/8), primes2_);
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
                uint64x2_t primes1;
                if (block_len == 1) {
                    primes1 = impl1;
                } else {
                    primes1 = vld1q_u64((uint64_t*)(primes.bits + idx1/8));
                }
                uint64_t impl_result;
                uint64x2_t primes_result;
                merge_neon_sp_single_register(i, impl1, primes1, &impl_result, &primes_result);

                vst1q_u64((uint64_t*)(primes.bits + idx1/8), primes_result);
                if (i >= first_difference) {
                    output_ptr[o_idx/64] = impl_result;
                    o_idx += 64;
                }
                idx1 += 128;
            }
        }
    }
}

#endif
