#pragma once
#ifdef __aarch64__
#include "bits_sp.h"
#include <arm_neon.h>
#include <assert.h>

static inline void merge_neon_sp_block(
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
    size_t o_idx64 = o_idx32;
    if (5 >= first_difference) { o_idx64 += (1 << (num_bits - 1)); }
    


    uint64_t *output_ptr = (uint64_t *) implicants.bits;

    size_t num_registers = (1 << num_bits) / 128;
    for (size_t regist = 0; regist < num_registers; regist++) {

        size_t idx1 = input_index + 128 * regist;
        uint64x2_t original_implicant = vld1q_u64((uint64_t*)(implicants.bits + idx1/8));
        uint64x2_t primes_implicant = original_implicant;

        uint64x2_t temp1;
        uint64x2_t temp2;
        uint64x2_t temp4;
        uint64x2_t temp8;
        uint64x2_t temp16;
        uint64x2_t temp32;
        uint64x2_t temp64;

        int64x2_t cnt1 = vdupq_n_s64(-1);  
        int64x2_t cnt2 = vdupq_n_s64(-2);
        int64x2_t cnt4 = vdupq_n_s64(-4);
        int64x2_t cnt8 = vdupq_n_s64(-8);
        int64x2_t cnt16 = vdupq_n_s64(-16);
        int64x2_t cnt32 = vdupq_n_s64(-32);

        temp1 = vshlq_u64(original_implicant, cnt1);
        temp2 = vshlq_u64(original_implicant, cnt2);
        temp4 = vshlq_u64(original_implicant, cnt4);
        temp8 = vshlq_u64(original_implicant, cnt8);
        temp16 = vshlq_u64(original_implicant, cnt16);
        temp32 = vshlq_u64(original_implicant, cnt32);
        temp64 = vextq_u64(original_implicant, vdupq_n_u64(0), 1);


        uint64x2_t aggregated1 = vandq_u64(original_implicant, temp1);
        uint64x2_t aggregated2 = vandq_u64(original_implicant, temp2);
        uint64x2_t aggregated4 = vandq_u64(original_implicant, temp4);
        uint64x2_t aggregated8 = vandq_u64(original_implicant, temp8);
        uint64x2_t aggregated16 = vandq_u64(original_implicant, temp16);
        uint64x2_t aggregated32 = vandq_u64(original_implicant, temp32);
        uint64x2_t aggregated64 = vandq_u64(original_implicant, temp64);

        uint64x2_t initial_result1 = vdupq_n_u64(0);
        uint64x2_t initial_result2 = vdupq_n_u64(0);
        uint64x2_t initial_result4 = vdupq_n_u64(0);
        uint64x2_t initial_result8 = vdupq_n_u64(0);
        uint64x2_t initial_result16 = vdupq_n_u64(0);
        uint64x2_t initial_result32 = vdupq_n_u64(0);
        uint64x2_t initial_result64 = vdupq_n_u64(0);

        uint64x2_t shifted1 = vdupq_n_u64(0);
        uint64x2_t shifted2 = vdupq_n_u64(0);
        uint64x2_t shifted4 = vdupq_n_u64(0);
        uint64x2_t shifted8 = vdupq_n_u64(0);
        uint64x2_t shifted16 = vdupq_n_u64(0);
        uint64x2_t shifted32 = vdupq_n_u64(0);
        uint64x2_t shifted64 = vdupq_n_u64(0);

        aggregated1 = vandq_u64(aggregated1, vdupq_n_u8(0b01010101));
        initial_result1 = aggregated1;
        shifted1 = vshrq_n_u64(aggregated1, 1);

        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vdupq_n_u8(0b00110011));
        aggregated2 = vandq_u64(aggregated2, vdupq_n_u8(0b00110011));
        initial_result2 = aggregated2;
        shifted1 = vshrq_n_u64(aggregated1, 2);
        shifted2 = vshrq_n_u64(aggregated2, 2);
    
        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vdupq_n_u8(0b00001111));
        aggregated2 = vandq_u64(vorrq_u64(aggregated2, shifted2), vdupq_n_u8(0b00001111));
        aggregated4 = vandq_u64(aggregated4, vdupq_n_u8(0b00001111));
        initial_result4 = aggregated4;
        shifted1 = vshrq_n_u64(aggregated1, 4);
        shifted2 = vshrq_n_u64(aggregated2, 4);
        shifted4 = vshrq_n_u64(aggregated4, 4);

        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vdupq_n_u16(0x00FF));
        aggregated2 = vandq_u64(vorrq_u64(aggregated2, shifted2), vdupq_n_u16(0x00FF));
        aggregated4 = vandq_u64(vorrq_u64(aggregated4, shifted4), vdupq_n_u16(0x00FF));
        aggregated8 = vandq_u64(aggregated8, vdupq_n_u16(0x00FF));
        initial_result8 = aggregated8;
        shifted1 = vshrq_n_u64(aggregated1, 8);
        shifted2 = vshrq_n_u64(aggregated2, 8);
        shifted4 = vshrq_n_u64(aggregated4, 8);
        shifted8 = vshrq_n_u64(aggregated8, 8);

        // 16 bit 
        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vdupq_n_u32(0x0000FFFF));
        aggregated2 = vandq_u64(vorrq_u64(aggregated2, shifted2), vdupq_n_u32(0x0000FFFF));
        aggregated4 = vandq_u64(vorrq_u64(aggregated4, shifted4), vdupq_n_u32(0x0000FFFF));
        aggregated8 = vandq_u64(vorrq_u64(aggregated8, shifted8), vdupq_n_u32(0x0000FFFF));
        aggregated16 = vandq_u64(aggregated16, vdupq_n_u32(0x0000FFFF));
        initial_result16 = aggregated16;
        shifted1 = vshrq_n_u64(aggregated1, 16);
        shifted2 = vshrq_n_u64(aggregated2, 16);
        shifted4 = vshrq_n_u64(aggregated4, 16);
        shifted8 = vshrq_n_u64(aggregated8, 16);
        shifted16 = vshrq_n_u64(aggregated16, 16);

        // 32 bit
        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vdupq_n_u64(0x00000000FFFFFFFF));
        aggregated2 = vandq_u64(vorrq_u64(aggregated2, shifted2), vdupq_n_u64(0x00000000FFFFFFFF));
        aggregated4 = vandq_u64(vorrq_u64(aggregated4, shifted4), vdupq_n_u64(0x00000000FFFFFFFF));
        aggregated8 = vandq_u64(vorrq_u64(aggregated8, shifted8), vdupq_n_u64(0x00000000FFFFFFFF));
        aggregated16 = vandq_u64(vorrq_u64(aggregated16, shifted16), vdupq_n_u64(0x00000000FFFFFFFF));
        aggregated32 = vandq_u64(aggregated32, vdupq_n_u64(0x00000000FFFFFFFF));
        initial_result32 = aggregated32;
    
        uint32x4_t tmp1 = vreinterpretq_u32_u64(aggregated1);
        uint32x4_t tmp2 = vreinterpretq_u32_u64(aggregated2);
        uint32x4_t tmp4 = vreinterpretq_u32_u64(aggregated4);
        uint32x4_t tmp8 = vreinterpretq_u32_u64(aggregated8);
        uint32x4_t tmp16 = vreinterpretq_u32_u64(aggregated16);
        uint32x4_t tmp32 = vreinterpretq_u32_u64(aggregated32);

        uint32x4_t tmp_shifted1 = vextq_u32(tmp1, tmp1, 1);
        uint32x4_t tmp_shifted2 = vextq_u32(tmp2, tmp2, 1);
        uint32x4_t tmp_shifted4 = vextq_u32(tmp4, tmp4, 1);
        uint32x4_t tmp_shifted8 = vextq_u32(tmp8, tmp8, 1);
        uint32x4_t tmp_shifted16 = vextq_u32(tmp16, tmp16, 1);
        uint32x4_t tmp_shifted32 = vextq_u32(tmp32, tmp32, 1);
        
        shifted1 = vreinterpretq_u64_u32(tmp_shifted1);
        shifted2 = vreinterpretq_u64_u32(tmp_shifted2);
        shifted4 = vreinterpretq_u64_u32(tmp_shifted4);
        shifted8 = vreinterpretq_u64_u32(tmp_shifted8);
        shifted16 = vreinterpretq_u64_u32(tmp_shifted16);
        shifted32 = vreinterpretq_u64_u32(tmp_shifted32);
        
        // 64 bit
        aggregated1 = vandq_u64(vorrq_u64(aggregated1, shifted1), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated2 = vandq_u64(vorrq_u64(aggregated2, shifted2), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated4 = vandq_u64(vorrq_u64(aggregated4, shifted4), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated8 = vandq_u64(vorrq_u64(aggregated8, shifted8), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated16 = vandq_u64(vorrq_u64(aggregated16, shifted16), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated32 = vandq_u64(vorrq_u64(aggregated32, shifted32), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        aggregated64 = vandq_u64(vorrq_u64(aggregated64, shifted64), vcombine_u64(vcreate_u64(0xFFFFFFFFFFFFFFFF), vcreate_u64(0x0)));
        initial_result64 = aggregated64;

        uint64_t out_result1 = vgetq_lane_u64(aggregated1, 0);
        uint64_t out_result2 = vgetq_lane_u64(aggregated2, 0); 
        uint64_t out_result4 = vgetq_lane_u64(aggregated4, 0); 
        uint64_t out_result8 = vgetq_lane_u64(aggregated8, 0);
        uint64_t out_result16 = vgetq_lane_u64(aggregated16, 0);
        uint64_t out_result32 = vgetq_lane_u64(aggregated32, 0);
        uint64_t out_result64 = vgetq_lane_u64(aggregated64, 0);


        uint64x2_t merged1;
        uint64x2_t merged2;
        uint64x2_t merged4;
        uint64x2_t merged8;
        uint64x2_t merged16;
        uint64x2_t merged32;
        uint64x2_t merged64;
        
        int64x2_t shift_amount1 = vdupq_n_s64(1);
        int64x2_t shift_amount2 = vdupq_n_s64(2);
        int64x2_t shift_amount4 = vdupq_n_s64(4);
        int64x2_t shift_amount8 = vdupq_n_s64(8);
        int64x2_t shift_amount16 = vdupq_n_s64(16);
        int64x2_t shift_amount32 = vdupq_n_s64(32);

        merged1 = vshlq_u64(initial_result1, shift_amount1);
        merged2 = vshlq_u64(initial_result2, shift_amount2);
        merged4 = vshlq_u64(initial_result4, shift_amount4);
        merged8 = vshlq_u64(initial_result8, shift_amount8);
        merged16 = vshlq_u64(initial_result16, shift_amount16);
        merged32 = vshlq_u64(initial_result32, shift_amount32);
        merged64 = vextq_u64(vdupq_n_u64(0), initial_result64, 1);


        merged1 = vorrq_u64(merged1, initial_result1);
        merged2 = vorrq_u64(merged2, initial_result2);
        merged4 = vorrq_u64(merged4, initial_result4);
        merged8 = vorrq_u64(merged8, initial_result8);
        merged16 = vorrq_u64(merged16, initial_result16);
        merged32 = vorrq_u64(merged32, initial_result32);
        merged64 = vorrq_u64(merged64, initial_result64);


        uint64x2_t primes1 = vbicq_u64(primes_implicant, merged1);
        uint64x2_t primes2 = vbicq_u64(primes1, merged2);
        uint64x2_t primes4 = vbicq_u64(primes2, merged4); 
        uint64x2_t primes8 = vbicq_u64(primes4, merged8);
        uint64x2_t primes16 = vbicq_u64(primes8, merged16);
        uint64x2_t primes32 = vbicq_u64(primes16, merged32);
        uint64x2_t primes64 = vbicq_u64(primes32, merged64);

        vst1q_u64((uint64_t*)(primes.bits + idx1/8), primes64);

        if (0 >= first_difference) {
            output_ptr[o_idx1/64] = out_result1;
            o_idx1 += 64;
            output_ptr[o_idx2/64] = out_result2;
            o_idx2 += 64;
            output_ptr[o_idx4/64] = out_result4;
            o_idx4 += 64;
            output_ptr[o_idx8/64] = out_result8;
            o_idx8 += 64;
            output_ptr[o_idx16/64] = out_result16;
            o_idx16 += 64;
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        else if (1 >= first_difference) {
            output_ptr[o_idx2/64] = out_result2;
            o_idx2 += 64;
            output_ptr[o_idx4/64] = out_result4;
            o_idx4 += 64;
            output_ptr[o_idx8/64] = out_result8;
            o_idx8 += 64;
            output_ptr[o_idx16/64] = out_result16;
            o_idx16 += 64;
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        else if (2 >= first_difference) {
            output_ptr[o_idx4/64] = out_result4;
            o_idx4 += 64;
            output_ptr[o_idx8/64] = out_result8;
            o_idx8 += 64;
            output_ptr[o_idx16/64] = out_result16;
            o_idx16 += 64;
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        else if (3 >= first_difference) {
            output_ptr[o_idx8/64] = out_result8;
            o_idx8 += 64;
            output_ptr[o_idx16/64] = out_result16;
            o_idx16 += 64;
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        else if (4 >= first_difference) {
            output_ptr[o_idx16/64] = out_result16;
            o_idx16 += 64;
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        else if (5 >= first_difference) {
            output_ptr[o_idx32/64] = out_result32;
            o_idx32 += 64;
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }   
        else if (6 >= first_difference) {
            output_ptr[o_idx64/64] = out_result64;
            o_idx64 += 64;
        }
        idx1 += 128;
    }

    size_t o_idx = o_idx64;



    for (int i = 7; i < num_bits; ++i) {
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
    }
}

#endif
