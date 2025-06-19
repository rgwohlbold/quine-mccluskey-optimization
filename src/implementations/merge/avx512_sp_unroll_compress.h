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
#include "pext_sp.h"
#include "../../debug.h"
#include <stdio.h>
#include <stdint.h>


static inline void merge_avx512_sp_unroll_compress_register(
    bitmap implicants, bitmap primes,
    size_t idx1,
    size_t* o_idx1, size_t* o_idx2, size_t* o_idx4, size_t* o_idx8,
    size_t* o_idx16, size_t* o_idx32, size_t* o_idx64, size_t* o_idx128, size_t* o_idx256,
    __m512i indices_sr_64_lane, __m512i indices_sr_128_lane, __m512i indices_sr_256_reg,
    __m512i indices_sr_64_reg, __m512i indices_sr_128_reg, __m512i indices_lower_and_lower,
    __m512i indices_sl_64_lane, __m512i indices_sl_128_lane,
    __m512i and_mask_1, __m512i and_mask_2, __m512i and_mask_4, __m512i and_mask_8,
    __m512i and_mask_16, __m512i and_mask_32, __m512i and_mask_64, __m512i and_mask_128, __m512i and_mask_256,
    __m512i zero, int first_difference) {

    __m512i impl1 = _mm512_load_si512((__m512i*)(implicants.bits + idx1 / 8));
    __m512i primes_0 = impl1;

    __m512i impl2_1 = _mm512_srli_epi64(impl1, 1);
    __m512i impl2_2 = _mm512_srli_epi64(impl1, 2);
    __m512i impl2_4 = _mm512_srli_epi64(impl1, 4);
    __m512i impl2_8 = _mm512_srli_epi64(impl1, 8);
    __m512i impl2_16 = _mm512_srli_epi64(impl1, 16);
    __m512i impl2_32 = _mm512_srli_epi64(impl1, 32);
    __m512i impl2_64 = _mm512_permutexvar_epi64(indices_sr_64_lane, impl1);
    __m512i impl2_128 = _mm512_permutexvar_epi64(indices_sr_128_lane, impl1);
    __m512i impl2_256 = _mm512_permutexvar_epi64(indices_sr_256_reg, impl1);

    __m512i aggregated_1 = _mm512_and_si512(impl1, impl2_1);
    __m512i aggregated_2 = _mm512_and_si512(impl1, impl2_2);
    __m512i aggregated_4 = _mm512_and_si512(impl1, impl2_4);
    __m512i aggregated_8 = _mm512_and_si512(impl1, impl2_8);
    __m512i aggregated_16 = _mm512_and_si512(impl1, impl2_16);
    __m512i aggregated_32 = _mm512_and_si512(impl1, impl2_32);
    __m512i aggregated_64 = _mm512_and_si512(impl1, impl2_64);
    __m512i aggregated_128 = _mm512_and_si512(impl1, impl2_128);
    __m512i aggregated_256 = _mm512_and_si512(impl1, impl2_256);

    __m512i initial_result_1 = _mm512_setzero_si512();
    __m512i initial_result_2 = _mm512_setzero_si512();
    __m512i initial_result_4 = _mm512_setzero_si512();
    __m512i initial_result_8 = _mm512_setzero_si512();
    __m512i initial_result_16 = _mm512_setzero_si512();
    __m512i initial_result_32 = _mm512_setzero_si512();
    __m512i initial_result_64 = _mm512_setzero_si512();
    __m512i initial_result_128 = _mm512_setzero_si512();
    __m512i initial_result_256 = _mm512_setzero_si512();

    __m512i shifted_1 = _mm512_setzero_si512();
    __m512i shifted_2 = _mm512_setzero_si512();
    __m512i shifted_4 = _mm512_setzero_si512();
    __m512i shifted_8 = _mm512_setzero_si512();
    __m512i shifted_16 = _mm512_setzero_si512();
    __m512i shifted_32 = _mm512_setzero_si512();
    __m512i shifted_64 = _mm512_setzero_si512();
    __m512i shifted_128 = _mm512_setzero_si512();


    aggregated_1 = _mm512_and_si512(aggregated_1, and_mask_1);
    initial_result_1 = aggregated_1;
    shifted_1 = _mm512_srli_epi64(aggregated_1, 1);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_2);
    aggregated_2 = _mm512_and_si512(aggregated_2, and_mask_2);
    initial_result_2 = aggregated_2;
    shifted_1 = _mm512_srli_epi64(aggregated_1, 2);
    shifted_2 = _mm512_srli_epi64(aggregated_2, 2);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_4);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_4);
    aggregated_4 = _mm512_and_si512(aggregated_4, and_mask_4);
    initial_result_4 = aggregated_4;
    shifted_1 = _mm512_srli_epi64(aggregated_1, 4);
    shifted_2 = _mm512_srli_epi64(aggregated_2, 4);
    shifted_4 = _mm512_srli_epi64(aggregated_4, 4);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_8);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_8);
    aggregated_4 = _mm512_and_si512(_mm512_or_si512(aggregated_4, shifted_4), and_mask_8);
    aggregated_8 = _mm512_and_si512(aggregated_8, and_mask_8);
    initial_result_8 = aggregated_8;
    shifted_1 = _mm512_srli_epi64(aggregated_1, 8);
    shifted_2 = _mm512_srli_epi64(aggregated_2, 8);
    shifted_4 = _mm512_srli_epi64(aggregated_4, 8);
    shifted_8 = _mm512_srli_epi64(aggregated_8, 8);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_16);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_16);
    aggregated_4 = _mm512_and_si512(_mm512_or_si512(aggregated_4, shifted_4), and_mask_16);
    aggregated_8 = _mm512_and_si512(_mm512_or_si512(aggregated_8, shifted_8), and_mask_16);
    aggregated_16 = _mm512_and_si512(aggregated_16, and_mask_16);
    initial_result_16 = aggregated_16;
    shifted_1 = _mm512_srli_epi64(aggregated_1, 16);
    shifted_2 = _mm512_srli_epi64(aggregated_2, 16);
    shifted_4 = _mm512_srli_epi64(aggregated_4, 16);
    shifted_8 = _mm512_srli_epi64(aggregated_8, 16);
    shifted_16 = _mm512_srli_epi64(aggregated_16, 16);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_32);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_32);
    aggregated_4 = _mm512_and_si512(_mm512_or_si512(aggregated_4, shifted_4), and_mask_32);
    aggregated_8 = _mm512_and_si512(_mm512_or_si512(aggregated_8, shifted_8), and_mask_32);
    aggregated_16 = _mm512_and_si512(_mm512_or_si512(aggregated_16, shifted_16), and_mask_32);
    aggregated_32 = _mm512_and_si512(aggregated_32, and_mask_32);
    initial_result_32 = aggregated_32;
    shifted_1 = _mm512_alignr_epi8(zero, aggregated_1, 4);
    shifted_2 = _mm512_alignr_epi8(zero, aggregated_2, 4);
    shifted_4 = _mm512_alignr_epi8(zero, aggregated_4, 4);
    shifted_8 = _mm512_alignr_epi8(zero, aggregated_8, 4);
    shifted_16 = _mm512_alignr_epi8(zero, aggregated_16, 4);
    shifted_32 = _mm512_alignr_epi8(zero, aggregated_32, 4);

    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_64);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_64);
    aggregated_4 = _mm512_and_si512(_mm512_or_si512(aggregated_4, shifted_4), and_mask_64);
    aggregated_8 = _mm512_and_si512(_mm512_or_si512(aggregated_8, shifted_8), and_mask_64);
    aggregated_16 = _mm512_and_si512(_mm512_or_si512(aggregated_16, shifted_16), and_mask_64);
    aggregated_32 = _mm512_and_si512(_mm512_or_si512(aggregated_32, shifted_32), and_mask_64);
    aggregated_64 = _mm512_and_si512(aggregated_64, and_mask_64);
    initial_result_64 = aggregated_64;
    shifted_1 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_1);
    shifted_2 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_2);
    shifted_4 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_4);
    shifted_8 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_8);
    shifted_16 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_16);
    shifted_32 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_32);
    shifted_64 = _mm512_permutexvar_epi64(indices_sr_64_reg, aggregated_64);


    aggregated_1 = _mm512_and_si512(_mm512_or_si512(aggregated_1, shifted_1), and_mask_128);
    aggregated_2 = _mm512_and_si512(_mm512_or_si512(aggregated_2, shifted_2), and_mask_128);
    aggregated_4 = _mm512_and_si512(_mm512_or_si512(aggregated_4, shifted_4), and_mask_128);
    aggregated_8 = _mm512_and_si512(_mm512_or_si512(aggregated_8, shifted_8), and_mask_128);
    aggregated_16 = _mm512_and_si512(_mm512_or_si512(aggregated_16, shifted_16), and_mask_128);
    aggregated_32 = _mm512_and_si512(_mm512_or_si512(aggregated_32, shifted_32), and_mask_128);
    aggregated_64 = _mm512_and_si512(_mm512_or_si512(aggregated_64, shifted_64), and_mask_128);
    aggregated_128 = _mm512_and_si512(aggregated_128, and_mask_128);
    initial_result_128 = aggregated_128;
    shifted_1 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_1);
    shifted_2 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_2);
    shifted_4 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_4);
    shifted_8 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_8);
    shifted_16 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_16);
    shifted_32 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_32);
    shifted_64 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_64);
    shifted_128 = _mm512_permutexvar_epi64(indices_sr_128_reg, aggregated_128);

    // AND with a mask is not necessary here, since we store the lower half of the result
    aggregated_1 = _mm512_or_si512(aggregated_1, shifted_1);
    aggregated_2 = _mm512_or_si512(aggregated_2, shifted_2);
    aggregated_4 = _mm512_or_si512(aggregated_4, shifted_4);
    aggregated_8 = _mm512_or_si512(aggregated_8, shifted_8);
    aggregated_16 = _mm512_or_si512(aggregated_16, shifted_16);
    aggregated_32 = _mm512_or_si512(aggregated_32, shifted_32);
    aggregated_64 = _mm512_or_si512(aggregated_64, shifted_64);
    aggregated_128 = _mm512_or_si512(aggregated_128, shifted_128);

    aggregated_256 = _mm512_and_si512(aggregated_256, and_mask_256);
    initial_result_256 = aggregated_256;


    switch(first_difference) {
    case 0:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx1) / 8), _mm512_castsi512_si256(aggregated_1));
        (*o_idx1) += 256;

    case 1:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx2) / 8), _mm512_castsi512_si256(aggregated_2));
        (*o_idx2) += 256;

    case 2:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx4) / 8), _mm512_castsi512_si256(aggregated_4));
        (*o_idx4) += 256;

    case 3:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx8) / 8), _mm512_castsi512_si256(aggregated_8));
        (*o_idx8) += 256;

    case 4:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx16) / 8), _mm512_castsi512_si256(aggregated_16));
        (*o_idx16) += 256;

    case 5:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx32) / 8), _mm512_castsi512_si256(aggregated_32));
        (*o_idx32) += 256;

    case 6:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx64) / 8), _mm512_castsi512_si256(aggregated_64));
        (*o_idx64) += 256;

    case 7:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx128) / 8), _mm512_castsi512_si256(aggregated_128));
        (*o_idx128) += 256;

    case 8:
        _mm256_store_si256((__m256i*)(implicants.bits + (*o_idx256) / 8), _mm512_castsi512_si256(aggregated_256));
        (*o_idx256) += 256;

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
}



static void merge_avx512_sp_unroll_compress(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    // Keep existing code for small inputs
    if (num_bits <= 8) {
        merge_pext_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    // Set up output indices as before
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

    // Initialize output indices based on first_difference (keep existing code)
    switch(first_difference) {
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




    __m512i indices_sr_64_lane = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1);
    __m512i indices_sr_128_lane = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2);
    __m512i indices_sr_64_reg =  _mm512_set_epi64(7, 7, 6, 5, 4, 3, 2, 1);
    __m512i indices_sr_128_reg =  _mm512_set_epi64(7, 7, 7, 6, 5, 4, 3, 2);
    __m512i indices_sr_256_reg =  _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
    __m512i indices_lower_and_lower =  _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);

    __m512i indices_sl_64_lane = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
    __m512i indices_sl_128_lane = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);

    __m512i and_mask_1 = _mm512_set1_epi8(0b01010101);
    __m512i and_mask_2 = _mm512_set1_epi8(0b00110011);
    __m512i and_mask_4 = _mm512_set1_epi8(0b00001111);
    __m512i and_mask_8 = _mm512_set1_epi16(0x00FF);
    __m512i and_mask_16 = _mm512_set1_epi32(0x0000FFFF);
    __m512i and_mask_32 = _mm512_set1_epi64(0x00000000FFFFFFFF);
    __m512i and_mask_64 = _mm512_set_epi64(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF);
    __m512i and_mask_128 = _mm512_set_epi64(0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    __m512i and_mask_256 = _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);

    __m512i zero = _mm512_setzero_si512();

    size_t num_registers = (1 << num_bits) / 512;
    size_t idx1 = input_index;

    // Process registers with increased ILP when possible
    if (num_registers >= 4) {
        // Process registers in groups of 4 for increased ILP
        for (size_t regist = 0; regist < num_registers; regist += 4) {

            size_t idx1_1 = idx1;
            size_t idx1_2 = idx1 + 512;
            size_t idx1_3 = idx1 + 1024;
            size_t idx1_4 = idx1 + 1536;

            // Process 4 registers in parallel to increase instruction-level parallelism
            merge_avx512_sp_unroll_compress_register(implicants, primes, idx1_1, &o_idx1, &o_idx2, &o_idx4, &o_idx8, &o_idx16, &o_idx32,
                              &o_idx64, &o_idx128, &o_idx256, indices_sr_64_lane, indices_sr_128_lane, indices_sr_256_reg,
                              indices_sr_64_reg, indices_sr_128_reg, indices_lower_and_lower, indices_sl_64_lane,
                              indices_sl_128_lane, and_mask_1, and_mask_2, and_mask_4, and_mask_8,
                              and_mask_16, and_mask_32, and_mask_64, and_mask_128, and_mask_256, zero, first_difference);



            merge_avx512_sp_unroll_compress_register(implicants, primes, idx1_2, &o_idx1, &o_idx2, &o_idx4, &o_idx8, &o_idx16, &o_idx32,
                                &o_idx64, &o_idx128, &o_idx256, indices_sr_64_lane, indices_sr_128_lane, indices_sr_256_reg,
                                indices_sr_64_reg, indices_sr_128_reg, indices_lower_and_lower, indices_sl_64_lane,
                                indices_sl_128_lane, and_mask_1, and_mask_2, and_mask_4, and_mask_8,
                                and_mask_16, and_mask_32, and_mask_64, and_mask_128, and_mask_256, zero, first_difference);



            merge_avx512_sp_unroll_compress_register(implicants, primes, idx1_3, &o_idx1, &o_idx2, &o_idx4, &o_idx8, &o_idx16, &o_idx32,
                                &o_idx64, &o_idx128, &o_idx256, indices_sr_64_lane, indices_sr_128_lane, indices_sr_256_reg,
                                indices_sr_64_reg, indices_sr_128_reg, indices_lower_and_lower, indices_sl_64_lane,
                                indices_sl_128_lane, and_mask_1, and_mask_2, and_mask_4, and_mask_8,
                                and_mask_16, and_mask_32, and_mask_64, and_mask_128, and_mask_256, zero, first_difference);



            merge_avx512_sp_unroll_compress_register(implicants, primes, idx1_4, &o_idx1, &o_idx2, &o_idx4, &o_idx8, &o_idx16, &o_idx32,
                                &o_idx64, &o_idx128, &o_idx256, indices_sr_64_lane, indices_sr_128_lane, indices_sr_256_reg,
                                indices_sr_64_reg, indices_sr_128_reg, indices_lower_and_lower, indices_sl_64_lane,
                                indices_sl_128_lane, and_mask_1, and_mask_2, and_mask_4, and_mask_8,
                                and_mask_16, and_mask_32, and_mask_64, and_mask_128, and_mask_256, zero, first_difference);

            idx1 += 4*512;
        }
    } else {
        // For smaller register counts, use the regular loop to avoid out-of-bounds access
        for (size_t regist = 0; regist < num_registers; regist++) {
            merge_avx512_sp_unroll_compress_register(implicants, primes, idx1, &o_idx1, &o_idx2, &o_idx4, &o_idx8, &o_idx16, &o_idx32,
                              &o_idx64, &o_idx128, &o_idx256, indices_sr_64_lane, indices_sr_128_lane, indices_sr_256_reg,
                              indices_sr_64_reg, indices_sr_128_reg, indices_lower_and_lower, indices_sl_64_lane,
                              indices_sl_128_lane, and_mask_1, and_mask_2, and_mask_4, and_mask_8,
                              and_mask_16, and_mask_32, and_mask_64, and_mask_128, and_mask_256, zero, first_difference);
            idx1 += 512;
        }
    }

    // Process larger block lengths as before
    size_t o_idx = output_index;
    if (first_difference <= 9) {
        o_idx += (9 - first_difference) * num_registers * 256;
    }

    // Keep existing code for larger block lengths
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
