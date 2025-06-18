#pragma once
#ifdef __aarch64__
#include "bits_sp.h"
#include <arm_neon.h>
#include <assert.h>
#define LOG_BLOCK_SIZE 1
#ifndef LOG_BLOCK_SIZE
#error "need to define LOG_BLOCK_SIZE"
#endif
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
        LOG_DEBUG("o_idx1: %zu, o_idx2: %zu, o_idx4: %zu, o_idx8: %zu, o_idx16: %zu, o_idx32: %zu, o_idx64: %zu", o_idx1, o_idx2, o_idx4, o_idx8, o_idx16, o_idx32, o_idx64);   

        idx1 += 128;
    }

    size_t o_idx = o_idx64;

    int i = 7;

    #if LOG_BLOCK_SIZE >= 2
    for (; i + 2 < num_bits; i += 3) {


        int  block_len  = 1 << i;
        int  num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 4) {
            size_t base = input_index + 2 * block * block_len;
            // the 8 starting offsets
            size_t idx0 = base;
            size_t idx1 = base +   block_len;
            size_t idx2 = base + 2*block_len;
            size_t idx3 = base + 3*block_len;
            size_t idx4 = base + 4*block_len;
            size_t idx5 = base + 5*block_len;
            size_t idx6 = base + 6*block_len;
            size_t idx7 = base + 7*block_len;

            for (int k = 0; k < block_len; k += 128) {
                // load implicants
                uint64x2_t impl0 = vld1q_u64((uint64_t*)(implicants.bits + (idx0)/8));
                uint64x2_t impl1 = vld1q_u64((uint64_t*)(implicants.bits + (idx1)/8));
                uint64x2_t impl2 = vld1q_u64((uint64_t*)(implicants.bits + (idx2)/8));
                uint64x2_t impl3 = vld1q_u64((uint64_t*)(implicants.bits + (idx3)/8));
                uint64x2_t impl4 = vld1q_u64((uint64_t*)(implicants.bits + (idx4)/8));
                uint64x2_t impl5 = vld1q_u64((uint64_t*)(implicants.bits + (idx5)/8));
                uint64x2_t impl6 = vld1q_u64((uint64_t*)(implicants.bits + (idx6)/8));
                uint64x2_t impl7 = vld1q_u64((uint64_t*)(implicants.bits + (idx7)/8));

                // compute all the pairwise ANDs we need for 3 levels of blocking
                uint64x2_t r01 = vandq_u64(impl0, impl1);
                uint64x2_t r23 = vandq_u64(impl2, impl3);
                uint64x2_t r45 = vandq_u64(impl4, impl5);
                uint64x2_t r67 = vandq_u64(impl6, impl7);

                uint64x2_t r02 = vandq_u64(impl0, impl2);
                uint64x2_t r13 = vandq_u64(impl1, impl3);
                uint64x2_t r46 = vandq_u64(impl4, impl6);
                uint64x2_t r57 = vandq_u64(impl5, impl7);

                uint64x2_t r04 = vandq_u64(impl0, impl4);
                uint64x2_t r15 = vandq_u64(impl1, impl5);
                uint64x2_t r26 = vandq_u64(impl2, impl6);
                uint64x2_t r37 = vandq_u64(impl3, impl7);

                // now mask out primes in three stages
                uint64x2_t p0 = vld1q_u64((uint64_t*)(primes.bits + (idx0)/8));
                uint64x2_t p1 = vld1q_u64((uint64_t*)(primes.bits + (idx1)/8));
                uint64x2_t p2 = vld1q_u64((uint64_t*)(primes.bits + (idx2)/8));
                uint64x2_t p3 = vld1q_u64((uint64_t*)(primes.bits + (idx3)/8));
                uint64x2_t p4 = vld1q_u64((uint64_t*)(primes.bits + (idx4)/8));
                uint64x2_t p5 = vld1q_u64((uint64_t*)(primes.bits + (idx5)/8));
                uint64x2_t p6 = vld1q_u64((uint64_t*)(primes.bits + (idx6)/8));
                uint64x2_t p7 = vld1q_u64((uint64_t*)(primes.bits + (idx7)/8));

                uint64x2_t m0 = vorrq_u64(r01, vorrq_u64(r02, r04));              
                uint64x2_t m1 = vorrq_u64(r01, vorrq_u64(r13, r15));
                uint64x2_t m2 = vorrq_u64(r23, vorrq_u64(r02, r26));
                uint64x2_t m3 = vorrq_u64(r23, vorrq_u64(r13, r37));
                uint64x2_t m4 = vorrq_u64(r45, vorrq_u64(r46, r04));
                uint64x2_t m5 = vorrq_u64(r45, vorrq_u64(r57, r15));
                uint64x2_t m6 = vorrq_u64(r67, vorrq_u64(r46, r26));
                uint64x2_t m7 = vorrq_u64(r67, vorrq_u64(r57, r37));

                p0 = vbicq_u64(p0, m0);
                p1 = vbicq_u64(p1, m1);
                p2 = vbicq_u64(p2, m2);
                p3 = vbicq_u64(p3, m3);
                p4 = vbicq_u64(p4, m4);
                p5 = vbicq_u64(p5, m5);
                p6 = vbicq_u64(p6, m6);
                p7 = vbicq_u64(p7, m7);


                // // first level
                // p0 = vbicq_u64(p0, r01);  p1 = vbicq_u64(p1, r01);
                // p2 = vbicq_u64(p2, r23);  p3 = vbicq_u64(p3, r23);
                // p4 = vbicq_u64(p4, r45);  p5 = vbicq_u64(p5, r45);
                // p6 = vbicq_u64(p6, r67);  p7 = vbicq_u64(p7, r67);

                // // second level
                // p0 = vbicq_u64(p0, r02);  p1 = vbicq_u64(p1, r13);
                // p2 = vbicq_u64(p2, r02);  p3 = vbicq_u64(p3, r13);
                // p4 = vbicq_u64(p4, r46);  p5 = vbicq_u64(p5, r57);
                // p6 = vbicq_u64(p6, r46);  p7 = vbicq_u64(p7, r57);

                // // third level
                // p0 = vbicq_u64(p0, r04);  p1 = vbicq_u64(p1, r15);
                // p2 = vbicq_u64(p2, r26);  p3 = vbicq_u64(p3, r37);
                // p4 = vbicq_u64(p4, r04);  p5 = vbicq_u64(p5, r15);
                // p6 = vbicq_u64(p6, r26);  p7 = vbicq_u64(p7, r37);

                // store back primes
                vst1q_u64((uint64_t*)(primes.bits + (idx0)/8), p0);
                vst1q_u64((uint64_t*)(primes.bits + (idx1)/8), p1);
                vst1q_u64((uint64_t*)(primes.bits + (idx2)/8), p2);
                vst1q_u64((uint64_t*)(primes.bits + (idx3)/8), p3);
                vst1q_u64((uint64_t*)(primes.bits + (idx4)/8), p4);
                vst1q_u64((uint64_t*)(primes.bits + (idx5)/8), p5);
                vst1q_u64((uint64_t*)(primes.bits + (idx6)/8), p6);
                vst1q_u64((uint64_t*)(primes.bits + (idx7)/8), p7);

                // emit up to three results
                if (i   >= first_difference) {
                    size_t out = output_index + ((i - first_difference) << (num_bits-1)) + (block * block_len + k);
                    vst1q_u64((uint64_t*)(implicants.bits + out     /8), r01);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+block_len)/8), r23);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+2*block_len)/8), r45);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+3*block_len)/8), r67);
                }
                if (i+1 >= first_difference) {
                    size_t out = output_index + ((i + 1 - first_difference) << (num_bits-1)) + (block * block_len + k);
                    vst1q_u64((uint64_t*)(implicants.bits + out     /8), r02);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+block_len)/8), r13);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+2*block_len)/8), r46);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+3*block_len)/8), r57);
                }
                if (i+2 >= first_difference) {
                    size_t out = output_index + ((i + 2 - first_difference) << (num_bits-1)) + (block * block_len + k);
                    vst1q_u64((uint64_t*)(implicants.bits + out     /8), r04);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+block_len)/8), r15);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+2*block_len)/8), r26);
                    vst1q_u64((uint64_t*)(implicants.bits + (out+3*block_len)/8), r37);
                }
                idx0 += 128;
                idx1 += 128;
                idx2 += 128;
                idx3 += 128;
                idx4 += 128;
                idx5 += 128;
                idx6 += 128;
                idx7 += 128;
            
            }
        }
    }
    #endif

    #if LOG_BLOCK_SIZE >= 1
    for (; i < num_bits - 1; i += 2) {
    
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);
        for (int block = 0; block < num_blocks; block += 2) {
            size_t idx0 = input_index + 2 * block * block_len;
            size_t idx1 = idx0 + block_len;
            size_t idx2 = idx1 + block_len;
            size_t idx3 = idx2 + block_len;
            
            for (int k = 0; k < block_len; k += 128) {
   
                uint64x2_t impl0 = vld1q_u64((uint64_t*)(implicants.bits + idx0/8));
                uint64x2_t impl1 = vld1q_u64((uint64_t*)(implicants.bits + idx1/8));
                uint64x2_t impl2 = vld1q_u64((uint64_t*)(implicants.bits + idx2/8));
                uint64x2_t impl3 = vld1q_u64((uint64_t*)(implicants.bits + idx3/8));
            
                
                uint64x2_t res01 = vandq_u64(impl0, impl1);
                uint64x2_t res23 = vandq_u64(impl2, impl3);
                uint64x2_t res02 = vandq_u64(impl0, impl2);
                uint64x2_t res13 = vandq_u64(impl1, impl3);

                uint64x2_t primes0 = vld1q_u64((uint64_t*)(primes.bits + idx0/8));
                uint64x2_t primes1 = vld1q_u64((uint64_t*)(primes.bits + idx1/8));
                uint64x2_t primes2 = vld1q_u64((uint64_t*)(primes.bits + idx2/8));
                uint64x2_t primes3 = vld1q_u64((uint64_t*)(primes.bits + idx3/8));

                uint64x2_t primes0_ = vbicq_u64(primes0, res01);
                uint64x2_t primes1_ = vbicq_u64(primes1, res01);
                uint64x2_t primes2_ = vbicq_u64(primes2, res23);
                uint64x2_t primes3_ = vbicq_u64(primes3, res23);

                uint64x2_t primes0__ = vbicq_u64(primes0_, res02);
                uint64x2_t primes1__ = vbicq_u64(primes1_, res13);
                uint64x2_t primes2__ = vbicq_u64(primes2_, res02);
                uint64x2_t primes3__ = vbicq_u64(primes3_, res13);

                vst1q_u64((uint64_t*)(primes.bits + idx0/8), primes0__);
                vst1q_u64((uint64_t*)(primes.bits + idx1/8), primes1__);
                vst1q_u64((uint64_t*)(primes.bits + idx2/8), primes2__);
                vst1q_u64((uint64_t*)(primes.bits + idx3/8), primes3__);

                if (i >= first_difference) {
                    size_t out = output_index + ((i - first_difference) << (num_bits-1)) + (block * block_len + k);
                    vst1q_u64((uint64_t*)(implicants.bits + out / 8), res01);
                    vst1q_u64((uint64_t*)(implicants.bits + (out + block_len) / 8), res23); 
              
            
                }
                if (i + 1 >= first_difference) {
                    size_t out = output_index + ((i + 1 - first_difference) << (num_bits - 1)) + (block * block_len + k);
                    vst1q_u64((uint64_t*)(implicants.bits + out / 8), res02);
                    vst1q_u64((uint64_t*)(implicants.bits + (out + block_len) / 8), res13);
                    
                }
                idx0 += 128;
                idx1 += 128;
                idx2 += 128;
                idx3 += 128;
             
            }
        }
    }
    o_idx = output_index + ((i - first_difference) << (num_bits - 1));
    #endif

    for (; i < num_bits; ++i) {
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
                        log_u64x2("saved", &res);
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
