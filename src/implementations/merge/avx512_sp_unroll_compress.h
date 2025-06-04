#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx512_single_pass"
#endif
#ifndef __AVX512F__
#error "need AVX512F for avx512_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>
#include "../../bitmap.h"
#include "bits_sp.h"
#include "../../debug.h"
#include <stdio.h>
#include <stdint.h>


static void merge_avx512_sp_unroll_compress(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    
    // LOG_INFO("first_difference=%d, num_bits=%d, input_index=%zu, output_index=%zu", first_difference, num_bits, input_index, output_index);
    
    // TODO: unroll this -> call merge_bits_spX directly
    if (num_bits <= 8) {
        merge_small_loop(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    // TODO: can ifs can be removed if higher o_idxes are not used?
    // try to do a switch - each case does all assignments? or something smarter if it's possible
    
    size_t chunk_offset = (1 << (num_bits - 1));


    size_t o_idx1 = output_index;
    size_t o_idx2 = output_index;
    size_t o_idx4 = output_index;
    size_t o_idx8 = output_index;
    size_t o_idx16 = output_index;
    size_t o_idx32 = output_index;  
    size_t o_idx64 = output_index;
    size_t o_idx128 = output_index;
    size_t o_idx256 = output_index; 

    switch(first_difference){
        case 0:
            o_idx2 += chunk_offset;
            o_idx4 += chunk_offset * 2;
            o_idx8 += chunk_offset * 3;
            o_idx16 += chunk_offset * 4;
            o_idx32 += chunk_offset * 5;
            o_idx64 += chunk_offset * 6;
            o_idx128 += chunk_offset * 7;
            o_idx256 += chunk_offset * 8;
            break;
        case 1:
            o_idx4 += chunk_offset;
            o_idx8 += chunk_offset * 2;
            o_idx16 += chunk_offset * 3;
            o_idx32 += chunk_offset * 4;
            o_idx64 += chunk_offset * 5;
            o_idx128 += chunk_offset * 6;
            o_idx256 += chunk_offset * 7;
            break;
        case 2:
            o_idx8 += chunk_offset;
            o_idx16 += chunk_offset * 2;
            o_idx32 += chunk_offset * 3;
            o_idx64 += chunk_offset * 4;
            o_idx128 += chunk_offset * 5;
            o_idx256 += chunk_offset * 6;
            break;
        case 3:
            o_idx16 += chunk_offset;
            o_idx32 += chunk_offset * 2;
            o_idx64 += chunk_offset * 3;
            o_idx128 += chunk_offset * 4;
            o_idx256 += chunk_offset * 5;
            break;
        case 4:
            o_idx32 += chunk_offset;
            o_idx64 += chunk_offset * 2;
            o_idx128 += chunk_offset * 3;
            o_idx256 += chunk_offset * 4;
            break;
        case 5:
            o_idx64 += chunk_offset;
            o_idx128 += chunk_offset * 2;
            o_idx256 += chunk_offset * 3;
            break;
        case 6:
            o_idx128 += chunk_offset;
            o_idx256 += chunk_offset * 2;
            break;
        case 7:
            o_idx256 += chunk_offset;
            break;
         
        default:
            break;
    }


    size_t num_registers = (1 << num_bits) / 512;
    size_t idx1 = input_index;

    __m512i indices_sr_64_lane = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1); 
    __m512i indices_sr_128_lane = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2);
    __m512i indices_sr_256_reg =  _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
    // __m512i indices_result_to_lower =  _mm512_set_epi64(7, 6, 5, 4, 5, 4, 1, 0);
    __m512i indices_lower_and_lower =  _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);

    __m512i indices_sl_64_lane = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0); 
    __m512i indices_sl_128_lane = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0); 

    // Create masks for different bit patterns
    __mmask8 mask_1 = 0x55;  // 01010101 - every other bit
    __mmask8 mask_2 = 0x33;  // 00110011 - every 2 bits
    __mmask8 mask_4 = 0x0F;  // 00001111 - every 4 bits
    __mmask8 mask_8 = 0x00FF;  // every 8 bits
    __mmask8 mask_16 = 0x00FF;  // Used with 16-bit grouping
    __mmask8 mask_32 = 0x00FF;  // Used with 32-bit grouping
    __mmask8 mask_64 = 0x55;   // Used with 64-bit grouping
    __mmask8 mask_128 = 0x33;  // Used with 128-bit grouping
    __mmask8 mask_256 = 0x0F;  // Used with 256-bit grouping

    // TODO: TRY WITH COMPRESS INSTEAD OF SHIFTING
    for (size_t regist = 0; regist < num_registers; regist++) {

        __m512i impl1 = _mm512_load_si512((__m512i*)(implicants.bits + idx1 / 8));
        __m512i primes_0 = impl1;

        // Step 1: Generate shifted versions for different block lengths
        __m512i impl2_1 = _mm512_srli_epi64(impl1, 1);
        __m512i impl2_2 = _mm512_srli_epi64(impl1, 2);
        __m512i impl2_4 = _mm512_srli_epi64(impl1, 4);
        __m512i impl2_8 = _mm512_srli_epi64(impl1, 8);
        __m512i impl2_16 = _mm512_srli_epi64(impl1, 16);
        __m512i impl2_32 = _mm512_srli_epi64(impl1, 32);
        __m512i impl2_64 = _mm512_permutexvar_epi64(indices_sr_64_lane, impl1);
        __m512i impl2_128 = _mm512_permutexvar_epi64(indices_sr_128_lane, impl1);
        __m512i impl2_256 = _mm512_permutexvar_epi64(indices_sr_256_reg, impl1);

        // Step 2: AND operations for each block length
        __m512i aggregated_1 = _mm512_and_si512(impl1, impl2_1);
        __m512i aggregated_2 = _mm512_and_si512(impl1, impl2_2);
        __m512i aggregated_4 = _mm512_and_si512(impl1, impl2_4);
        __m512i aggregated_8 = _mm512_and_si512(impl1, impl2_8);
        __m512i aggregated_16 = _mm512_and_si512(impl1, impl2_16);
        __m512i aggregated_32 = _mm512_and_si512(impl1, impl2_32);
        __m512i aggregated_64 = _mm512_and_si512(impl1, impl2_64);
        __m512i aggregated_128 = _mm512_and_si512(impl1, impl2_128);
        __m512i aggregated_256 = _mm512_and_si512(impl1, impl2_256);

        // Step 3: Use compress operations to aggregate bits
        // For block_len = 1
        __m512i initial_result_1 = _mm512_maskz_compress_epi64(mask_1, aggregated_1);
        __m512i compressed_1 = _mm512_mask_compress_epi64(_mm512_setzero_si512(), mask_1, aggregated_1);

        // For block_len = 2
        __m512i initial_result_2 = _mm512_maskz_compress_epi64(mask_2, aggregated_2);
        __m512i compressed_2 = _mm512_mask_compress_epi64(_mm512_setzero_si512(), mask_2, aggregated_2);

        // For block_len = 4
        __m512i initial_result_4 = _mm512_maskz_compress_epi64(mask_4, aggregated_4);
        __m512i compressed_4 = _mm512_mask_compress_epi64(_mm512_setzero_si512(), mask_4, aggregated_4);

        // For block_len = 8
        __m512i initial_result_8 = _mm512_maskz_compress_epi64(mask_8, aggregated_8);
        __m512i compressed_8 = _mm512_mask_compress_epi64(_mm512_setzero_si512(), mask_8, aggregated_8);

        // For block_len = 16
        __m512i initial_result_16 = aggregated_16;
        __m512i compressed_16 = _mm512_maskz_compress_epi16(mask_16, aggregated_16);

        // For block_len = 32
        __m512i initial_result_32 = aggregated_32;
        __m512i compressed_32 = _mm512_maskz_compress_epi32(mask_32, aggregated_32);

        // For block_len = 64
        __m512i initial_result_64 = aggregated_64;
        __m512i compressed_64 = _mm512_maskz_compress_epi64(mask_64, aggregated_64);

        // For block_len = 128
        __m512i initial_result_128 = aggregated_128;
        __m512i compressed_128 = _mm512_maskz_compress_epi64(mask_128, aggregated_128);

        // For block_len = 256
        __m512i initial_result_256 = aggregated_256;
        __m512i compressed_256 = _mm512_maskz_compress_epi64(mask_256, aggregated_256);

        // Step 4: Get final results for each block length
        aggregated_1 = compressed_1;
        aggregated_2 = compressed_2;
        aggregated_4 = compressed_4;
        aggregated_8 = compressed_8;
        aggregated_16 = compressed_16;
        aggregated_32 = compressed_32;
        aggregated_64 = compressed_64;
        aggregated_128 = compressed_128;
        aggregated_256 = compressed_256;


        switch(first_difference) {
        case 0:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx1 / 8), _mm512_castsi512_si256(aggregated_1));
            o_idx1 += 256;

        case 1:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx2 / 8), _mm512_castsi512_si256(aggregated_2));
            o_idx2 += 256;
            
        case 2:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx4 / 8), _mm512_castsi512_si256(aggregated_4));
            o_idx4 += 256;

        case 3:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx8 / 8), _mm512_castsi512_si256(aggregated_8));
            o_idx8 += 256;

        case 4:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx16 / 8), _mm512_castsi512_si256(aggregated_16));
            o_idx16 += 256;
            
        case 5:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx32 / 8), _mm512_castsi512_si256(aggregated_32));
            o_idx32 += 256;
            
        case 6:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx64 / 8), _mm512_castsi512_si256(aggregated_64));
            o_idx64 += 256;
    
        case 7:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx128 / 8), _mm512_castsi512_si256(aggregated_128));
            o_idx128 += 256;
            
        case 8:
            _mm256_store_si256((__m256i*)(implicants.bits + o_idx256 / 8), _mm512_castsi512_si256(aggregated_256));
            o_idx256 += 256;

        default:
            break;
        }
        
        // merged = initial_result_X | (initial_result_X << block_len)
        __m512i merged_1 = _mm512_or_si512(initial_result_1, _mm512_slli_epi64(initial_result_1, 1));
        __m512i merged_2 = _mm512_or_si512(initial_result_2, _mm512_slli_epi64(initial_result_2, 2));
        __m512i merged_4 = _mm512_or_si512(initial_result_4, _mm512_slli_epi64(initial_result_4, 4));
        __m512i merged_8 = _mm512_or_si512(initial_result_8, _mm512_slli_epi64(initial_result_8, 8));
        __m512i merged_16 = _mm512_or_si512(initial_result_16, _mm512_slli_epi64(initial_result_16, 16));
        __m512i merged_32 = _mm512_or_si512(initial_result_32, _mm512_slli_epi64(initial_result_32, 32));
        __m512i merged_64 = _mm512_permutexvar_epi64(indices_sl_64_lane, initial_result_64);
        __m512i merged_128 = _mm512_permutexvar_epi64(indices_sl_128_lane, initial_result_128);
        __m512i merged_256 = _mm512_permutexvar_epi64(indices_lower_and_lower, initial_result_256);
        
        
        __m512i primes_1 = _mm512_andnot_si512(merged_1, primes_0);
        __m512i primes_2 = _mm512_andnot_si512(merged_2, primes_1);
        __m512i primes_4 = _mm512_andnot_si512(merged_4, primes_2);
        __m512i primes_8 = _mm512_andnot_si512(merged_8, primes_4);
        __m512i primes_16 = _mm512_andnot_si512(merged_16, primes_8);
        __m512i primes_32 = _mm512_andnot_si512(merged_32, primes_16);
        __m512i primes_64 = _mm512_andnot_si512(merged_64, primes_32);
        __m512i primes_128 = _mm512_andnot_si512(merged_128, primes_64);
        __m512i primes_256 = _mm512_andnot_si512(merged_256, primes_128);

        _mm512_store_si512((__m512i*)(primes.bits + idx1 / 8), primes_256);

        idx1 += 512;
    }

    // implicants do not fit into one register, and we use the largest register size
    size_t o_idx = output_index;
    if (first_difference <= 9) {
        o_idx += (9 - first_difference) * num_registers * 256;
    }
    for (int i = 9; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            // TODO: try to unroll this loop
            for (int k = 0; k < block_len; k += 512) {
                __m512i impl1 = _mm512_load_si512((__m512i*)(implicants.bits + idx1 / 8));
                __m512i impl2 = _mm512_load_si512((__m512i*)(implicants.bits + idx2 / 8));
                __m512i primes1 = _mm512_load_si512((__m512i*)(primes.bits + idx1 / 8));
                __m512i primes2 = _mm512_load_si512((__m512i*)(primes.bits + idx2 / 8));
                __m512i res = _mm512_and_si512(impl1, impl2);
                __m512i primes1_ = _mm512_andnot_si512(res, primes1);
                __m512i primes2_ = _mm512_andnot_si512(res, primes2);
                _mm512_store_si512((__m512i*)(primes.bits + idx1 / 8), primes1_);
                _mm512_store_si512((__m512i*)(primes.bits + idx2 / 8), primes2_);
                if (i >= first_difference) {
                    _mm512_store_si512((__m512i*)(implicants.bits + o_idx / 8), res);
                    o_idx += 512;
                }
                idx1 += 512;
                idx2 += 512;
            }
        }
    }

}
