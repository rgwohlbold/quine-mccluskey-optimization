#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx512_single_pass"
#endif

#ifndef __AVX512F__
#error "need AVX512F for avx512_single_pass"
#endif

#ifndef LOG_BLOCK_SIZE_AVX512
#error "need to define LOG_BLOCK_SIZE_AVX512"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "./avx2_sp_ssa.h"
#include "../../debug.h"

static inline void merge_avx512_sp_single_register_shuffle_1(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 1;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation step
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

    // block_len == 1
    __m512i masked0 = _mm512_and_si512(aggregated0, _mm512_set1_epi8(0b01010101));
    __m512i shifted0 = _mm512_srli_epi64(masked0, 1);

    // block_len <= 2
    __m512i combined0 = _mm512_or_si512(masked0, shifted0);
    __m512i masked1 = _mm512_and_si512(combined0, _mm512_set1_epi8(0b00110011));
    __m512i shifted1 = _mm512_srli_epi64(masked1, 2);

    // block_len <= 4
    __m512i combined1 = _mm512_or_si512(masked1, shifted1);
    __m512i masked2 = _mm512_and_si512(combined1, _mm512_set1_epi8(0b00001111));
    __m512i shifted2 = _mm512_srli_epi64(masked2, 4);

    // block_len <= 8
    __m512i combined2 = _mm512_or_si512(masked2, shifted2);
    __m512i masked3 = _mm512_and_si512(combined2, _mm512_set1_epi16(0x00FF));

    // Shuffle bytes to extract the result
    // In AVX512 we need a different approach to shuffle
    // TODO: try shuffle with _mm512_shuffle_epi8
    __m512i shuffle_mask = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Upper 16 bytes (not used)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Upper-mid 16 bytes (not used)
        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,     // Lower-mid 16 bytes
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0           // Lower 16 bytes
    );
    __mmask64 k = 0x00000000FFFFFFFF;  // Lower 32 bytes are 1's, upper 32 bytes are 0's
    __m512i aggregated_final = _mm512_maskz_permutexvar_epi8(k, shuffle_mask, masked3);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    // shift initial_result left by block_len
    __m512i shifted_initial_result = _mm512_slli_epi64(masked0, block_len);
    __m512i merged = _mm512_or_si512(shifted_initial_result, masked0);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx512_sp_single_register_shuffle_2(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 2;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation step
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

    // Apply mask for block_len == 2
    __m512i masked0 = _mm512_and_si512(aggregated0, _mm512_set1_epi8(0b00110011));
    __m512i initial_result = masked0;
    __m512i shifted0 = _mm512_srli_epi64(masked0, 2);

    // Block aggregation for length <= 4
    __m512i combined0 = _mm512_or_si512(masked0, shifted0);
    __m512i masked1 = _mm512_and_si512(combined0, _mm512_set1_epi8(0b00001111));
    __m512i shifted1 = _mm512_srli_epi64(masked1, 4);

    // Block aggregation for length <= 8
    __m512i combined1 = _mm512_or_si512(masked1, shifted1);
    __m512i masked2 = _mm512_and_si512(combined1, _mm512_set1_epi16(0x00FF));

    // Shuffle bytes to extract the result
    __m512i shuffle_mask = _mm512_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi8(shuffle_mask, masked2);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    // Shift initial_result left by block_len
    __m512i shifted_initial_result = _mm512_slli_epi64(initial_result, block_len);
    __m512i merged = _mm512_or_si512(shifted_initial_result, initial_result);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_4(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 4;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

    // Apply mask for block_len == 4
    __m512i masked0 = _mm512_and_si512(aggregated0, _mm512_set1_epi8(0b00001111));
    __m512i initial_result = masked0;
    __m512i shifted0 = _mm512_srli_epi64(masked0, 4);

    // Block aggregation for length <= 8
    __m512i combined0 = _mm512_or_si512(masked0, shifted0);
    __m512i masked1 = _mm512_and_si512(combined0, _mm512_set1_epi16(0x00FF));

    // Shuffle bytes to extract the result
    __m512i shuffle_mask = _mm512_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi8(shuffle_mask, masked1);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    // Shift initial_result left by block_len
    __m512i shifted_initial_result = _mm512_slli_epi64(initial_result, block_len);
    __m512i merged = _mm512_or_si512(shifted_initial_result, initial_result);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_8(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 8;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

    // Apply mask for block_len == 8
    __m512i masked0 = _mm512_and_si512(aggregated0, _mm512_set1_epi16(0x00FF));
    __m512i initial_result = masked0;

    // Shuffle bytes to extract the result
    __m512i shuffle_mask = _mm512_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi8(shuffle_mask, masked0);

    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    // Shift initial_result left by block_len
    __m512i shifted_initial_result = _mm512_slli_epi64(initial_result, block_len);
    __m512i merged = _mm512_or_si512(shifted_initial_result, initial_result);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_16(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 16;

    // Block_len <= 64
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

    // Put the result into lower 256 bytes
    __m512i shuffle_mask = _mm512_set_epi16(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi16(shuffle_mask, aggregated0);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    __m512i prime_shuffle_mask = _mm512_set_epi16(
        30, 30, 28, 28,
        26, 26, 24, 24,
        22, 22, 20, 20,
        18, 18, 16, 16,
        14, 14, 12, 12,
        10, 10, 8, 8,
        6, 6, 4, 4,
        2, 2, 0, 0
    );
    __m512i merged = _mm512_permutexvar_epi16(prime_shuffle_mask, aggregated0);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_32(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    const int block_len = 32;

    // Block_len <= 64
    __m512i impl2 = _mm512_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);


    // Put the result into lower 256 bytes
    __m512i shuffle_mask = _mm512_set_epi32(
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi32(shuffle_mask, aggregated0);


    __m512i prime_shuffle_mask = _mm512_set_epi32(
        14, -1, 12, -1, 10, -1, 8, -1,
        6, -1, 4, -1, 2, -1, 0, -1
    );
    __m512i shifted_initial_result = _mm512_permutexvar_epi32(prime_shuffle_mask, aggregated0);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    // // Shift initial_result left by block_len
    // __m512i shifted_initial_result = _mm512_slli_epi64(aggregated0, block_len);
    __m512i merged = _mm512_or_si512(shifted_initial_result, aggregated0);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_64(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    // For 64-bit shifts within 512-bit register, use permutexvar
    __m512i indices = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1);
    __m512i impl2 = _mm512_permutexvar_epi64(indices, impl1);

    // Initial aggregation
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2);

   // Put the result into lower 256 bytes
    __m512i shuffle_mask = _mm512_set_epi64(
        -1, -1, -1, -1,
        6, 4, 2, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi64(shuffle_mask, aggregated0);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    __m512i merge_mask = _mm512_set_epi64(
        6, 6, 4, 4, 2, 2, 0, 0
    );
    __m512i merged = _mm512_permutexvar_epi64(merge_mask, aggregated0);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_128(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {

    __m512i indices_sr_128_lane = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2);
    __m512i impl2_128 = _mm512_permutexvar_epi64(indices_sr_128_lane, impl1);

    // AND operation between original and shuffled
    __m512i aggregated0 = _mm512_and_si512(impl1, impl2_128);

   // Put the result into lower 256 bytes
    __m512i shuffle_mask = _mm512_set_epi64(
        -1, -1, -1, -1,
        5, 4, 1, 0
    );
    __m512i aggregated_final = _mm512_permutexvar_epi64(shuffle_mask, aggregated0);


    // Extract lower 256 bits for result
    *result = _mm512_castsi512_si256(aggregated_final);

    __m512i merge_mask = _mm512_set_epi64(
         5, 4, 5, 4, 1, 0, 1, 0
    );
    __m512i merged = _mm512_permutexvar_epi64(merge_mask, aggregated0);
    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx512_sp_single_register_shuffle_256(__m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {

    __m512i indices_sr_256_reg =  _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
    __m512i impl2_256 = _mm512_permutexvar_epi64(indices_sr_256_reg, impl1);
    __m512i aggregated = _mm512_and_si512(impl1, impl2_256);

    // Return the result
    *result = _mm512_castsi512_si256(aggregated);

    // Broadcast result to both halves of a 512-bit register
    __m512i indices_lower_and_lower =  _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);
    __m512i merged_256 = _mm512_permutexvar_epi64(indices_lower_and_lower, aggregated);

    // Calculate primes result
    *primes_result = _mm512_andnot_si512(merged_256, primes1);

}


static inline void merge_avx512_sp_single_register_shuffle(int bit_difference, __m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    int block_len = 1 << bit_difference;

    switch (block_len)
    {
    case 1:
        merge_avx512_sp_single_register_shuffle_1(impl1, primes1, result, primes_result);
        break;
    case 2:
        merge_avx512_sp_single_register_shuffle_2(impl1, primes1, result, primes_result);
        break;
    case 4:
        merge_avx512_sp_single_register_shuffle_4(impl1, primes1, result, primes_result);
        break;
    case 8:
        merge_avx512_sp_single_register_shuffle_8(impl1, primes1, result, primes_result);
        break;
    case 16:
        merge_avx512_sp_single_register_shuffle_16(impl1, primes1, result, primes_result);
        break;
    case 32:
        merge_avx512_sp_single_register_shuffle_32(impl1, primes1, result, primes_result);
        break;
    case 64:
        merge_avx512_sp_single_register_shuffle_64(impl1, primes1, result, primes_result);
        break;
    case 128:
        merge_avx512_sp_single_register_shuffle_128(impl1, primes1, result, primes_result);
        break;
    case 256:
        merge_avx512_sp_single_register_shuffle_256(impl1, primes1, result, primes_result);
        break;
    default:
        break;
    }

}


static void merge_avx512_sp_block(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                                 int first_difference) {
    if (num_bits <= 8) {
        merge_avx2_sp_small_n_ssa(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    // For AVX512, the register size is 512 bits (64 bytes) instead of 256 bits
    int num_registers = (1 << (num_bits - 9));
    if (num_registers < 4) {
        for (int register_index = 0; register_index < num_registers; register_index += 1) {
            size_t idx1 = input_index + 512 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 32 * register_index; // 32 bytes = 256 bits

            __m512i impl1 = _mm512_load_si512((__m512i *)(implicants.bits + idx1 / 8));
            __m512i primes1 = impl1;
            for (int i = 0; i < 9; i++) {
                __m256i impl_result;
                __m512i primes_result;

                // Call the corresponding AVX512 helper function
                merge_avx512_sp_single_register_shuffle(i, impl1, primes1, &impl_result, &primes_result);
                primes1 = primes_result;
                if (i >= first_difference) {
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx1_b), impl_result);
                    o_idx1_b += 32 * num_registers;
                }
            }
            _mm512_store_si512((__m512i *)(primes.bits + idx1 / 8), primes1);
        }
        if (first_difference <= 9) {
            o_idx += (9 - first_difference) * num_registers * 256; // 256 bits output per iteration
        }
    } else {
        for (int register_index = 0; register_index < num_registers; register_index += 4) {
            size_t idx1 = input_index + 512 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 32 * register_index;

            __m512i impl1 = _mm512_load_si512((__m512i *)(implicants.bits + idx1 / 8));
            __m512i primes1 = impl1;
            __m512i impl2 = _mm512_load_si512((__m512i *)(implicants.bits + (idx1 + 512) / 8));
            __m512i primes2 = impl2;
            __m512i impl3 = _mm512_load_si512((__m512i *)(implicants.bits + (idx1 + 1024) / 8));
            __m512i primes3 = impl3;
            __m512i impl4 = _mm512_load_si512((__m512i *)(implicants.bits + (idx1 + 1536) / 8));
            __m512i primes4 = impl4;

            __m256i impl1_result, impl2_result, impl3_result, impl4_result;

            merge_avx512_sp_single_register_shuffle_1(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_1(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_1(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_1(impl4, primes4, &impl4_result, &primes4);
            if (0 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_2(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_2(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_2(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_2(impl4, primes4, &impl4_result, &primes4);
            if (1 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_4(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_4(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_4(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_4(impl4, primes4, &impl4_result, &primes4);
            if (2 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_8(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_8(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_8(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_8(impl4, primes4, &impl4_result, &primes4);
            if (3 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_16(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_16(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_16(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_16(impl4, primes4, &impl4_result, &primes4);
            if (4 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_32(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_32(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_32(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_32(impl4, primes4, &impl4_result, &primes4);
            if (5 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_64(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_64(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_64(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_64(impl4, primes4, &impl4_result, &primes4);
            if (6 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            merge_avx512_sp_single_register_shuffle_128(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_128(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_128(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_128(impl4, primes4, &impl4_result, &primes4);
            if (7 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            // New AVX512 shuffle functions can be added here for 256-bit blocks
            merge_avx512_sp_single_register_shuffle_256(impl1, primes1, &impl1_result, &primes1);
            merge_avx512_sp_single_register_shuffle_256(impl2, primes2, &impl2_result, &primes2);
            merge_avx512_sp_single_register_shuffle_256(impl3, primes3, &impl3_result, &primes3);
            merge_avx512_sp_single_register_shuffle_256(impl4, primes4, &impl4_result, &primes4);
            if (8 >= first_difference) {
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 32)), impl2_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 64)), impl3_result);
                _mm256_store_si256((__m256i *)(implicants.bits + (o_idx1_b + 96)), impl4_result);
                o_idx1_b += 32 * num_registers;
            }

            _mm512_store_si512((__m512i *)(primes.bits + idx1 / 8), primes1);
            _mm512_store_si512((__m512i *)(primes.bits + (idx1 + 512) / 8), primes2);
            _mm512_store_si512((__m512i *)(primes.bits + (idx1 + 1024) / 8), primes3);
            _mm512_store_si512((__m512i *)(primes.bits + (idx1 + 1536) / 8), primes4);
        }
        if (first_difference <= 9) { // Increased to 9 since we added 256-bit block processing
            o_idx += (9 - first_difference) * num_registers * 256;
        }
    }

    int i = 9; // Start from 9 since we processed 8 shuffle functions plus the 256-bit one
    const size_t input_index_b = input_index / 8;
    const size_t output_index_b = output_index / 8;


#if LOG_BLOCK_SIZE_AVX512 >= 3
    for (; i+3 < num_bits; i += 4) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 8) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 64) { // Process 64 bytes at a time with AVX512
                // Load 16 implicant chunks using AVX512
                __m512i impl0  = _mm512_load_si512((__m512i *)(impl_index0 +  0 * block_len_b));
                __m512i impl1  = _mm512_load_si512((__m512i *)(impl_index0 +  1 * block_len_b));
                __m512i impl2  = _mm512_load_si512((__m512i *)(impl_index0 +  2 * block_len_b));
                __m512i impl3  = _mm512_load_si512((__m512i *)(impl_index0 +  3 * block_len_b));
                __m512i impl4  = _mm512_load_si512((__m512i *)(impl_index0 +  4 * block_len_b));
                __m512i impl5  = _mm512_load_si512((__m512i *)(impl_index0 +  5 * block_len_b));
                __m512i impl6  = _mm512_load_si512((__m512i *)(impl_index0 +  6 * block_len_b));
                __m512i impl7  = _mm512_load_si512((__m512i *)(impl_index0 +  7 * block_len_b));
                __m512i impl8  = _mm512_load_si512((__m512i *)(impl_index0 +  8 * block_len_b));
                __m512i impl9  = _mm512_load_si512((__m512i *)(impl_index0 +  9 * block_len_b));
                __m512i impl10 = _mm512_load_si512((__m512i *)(impl_index0 + 10 * block_len_b));
                __m512i impl11 = _mm512_load_si512((__m512i *)(impl_index0 + 11 * block_len_b));
                __m512i impl12 = _mm512_load_si512((__m512i *)(impl_index0 + 12 * block_len_b));
                __m512i impl13 = _mm512_load_si512((__m512i *)(impl_index0 + 13 * block_len_b));
                __m512i impl14 = _mm512_load_si512((__m512i *)(impl_index0 + 14 * block_len_b));
                __m512i impl15 = _mm512_load_si512((__m512i *)(impl_index0 + 15 * block_len_b));

                // Load 16 prime chunks using AVX512
                __m512i primes0  = _mm512_load_si512((__m512i *)(primes_index0 +  0 * block_len_b));
                __m512i primes1  = _mm512_load_si512((__m512i *)(primes_index0 +  1 * block_len_b));
                __m512i primes2  = _mm512_load_si512((__m512i *)(primes_index0 +  2 * block_len_b));
                __m512i primes3  = _mm512_load_si512((__m512i *)(primes_index0 +  3 * block_len_b));
                __m512i primes4  = _mm512_load_si512((__m512i *)(primes_index0 +  4 * block_len_b));
                __m512i primes5  = _mm512_load_si512((__m512i *)(primes_index0 +  5 * block_len_b));
                __m512i primes6  = _mm512_load_si512((__m512i *)(primes_index0 +  6 * block_len_b));
                __m512i primes7  = _mm512_load_si512((__m512i *)(primes_index0 +  7 * block_len_b));
                __m512i primes8  = _mm512_load_si512((__m512i *)(primes_index0 +  8 * block_len_b));
                __m512i primes9  = _mm512_load_si512((__m512i *)(primes_index0 +  9 * block_len_b));
                __m512i primes10 = _mm512_load_si512((__m512i *)(primes_index0 + 10 * block_len_b));
                __m512i primes11 = _mm512_load_si512((__m512i *)(primes_index0 + 11 * block_len_b));
                __m512i primes12 = _mm512_load_si512((__m512i *)(primes_index0 + 12 * block_len_b));
                __m512i primes13 = _mm512_load_si512((__m512i *)(primes_index0 + 13 * block_len_b));
                __m512i primes14 = _mm512_load_si512((__m512i *)(primes_index0 + 14 * block_len_b));
                __m512i primes15 = _mm512_load_si512((__m512i *)(primes_index0 + 15 * block_len_b));

                // Compute results for all 4 levels using AVX512 intrinsics
                // Level i (distance 1)
                __m512i res_i_01   = _mm512_and_si512(impl0, impl1);
                __m512i res_i_23   = _mm512_and_si512(impl2, impl3);
                __m512i res_i_45   = _mm512_and_si512(impl4, impl5);
                __m512i res_i_67   = _mm512_and_si512(impl6, impl7);
                __m512i res_i_89   = _mm512_and_si512(impl8, impl9);
                __m512i res_i_1011 = _mm512_and_si512(impl10, impl11);
                __m512i res_i_1213 = _mm512_and_si512(impl12, impl13);
                __m512i res_i_1415 = _mm512_and_si512(impl14, impl15);

                // Level i+1 (distance 2)
                __m512i res_i1_02   = _mm512_and_si512(impl0, impl2);
                __m512i res_i1_13   = _mm512_and_si512(impl1, impl3);
                __m512i res_i1_46   = _mm512_and_si512(impl4, impl6);
                __m512i res_i1_57   = _mm512_and_si512(impl5, impl7);
                __m512i res_i1_810  = _mm512_and_si512(impl8, impl10);
                __m512i res_i1_911  = _mm512_and_si512(impl9, impl11);
                __m512i res_i1_1214 = _mm512_and_si512(impl12, impl14);
                __m512i res_i1_1315 = _mm512_and_si512(impl13, impl15);

                // Level i+2 (distance 4)
                __m512i res_i2_04   = _mm512_and_si512(impl0, impl4);
                __m512i res_i2_15   = _mm512_and_si512(impl1, impl5);
                __m512i res_i2_26   = _mm512_and_si512(impl2, impl6);
                __m512i res_i2_37   = _mm512_and_si512(impl3, impl7);
                __m512i res_i2_812  = _mm512_and_si512(impl8, impl12);
                __m512i res_i2_913  = _mm512_and_si512(impl9, impl13);
                __m512i res_i2_1014 = _mm512_and_si512(impl10, impl14);
                __m512i res_i2_1115 = _mm512_and_si512(impl11, impl15);

                // Level i+3 (distance 8)
                __m512i res_i3_08  = _mm512_and_si512(impl0, impl8);
                __m512i res_i3_19  = _mm512_and_si512(impl1, impl9);
                __m512i res_i3_210 = _mm512_and_si512(impl2, impl10);
                __m512i res_i3_311 = _mm512_and_si512(impl3, impl11);
                __m512i res_i3_412 = _mm512_and_si512(impl4, impl12);
                __m512i res_i3_513 = _mm512_and_si512(impl5, impl13);
                __m512i res_i3_614 = _mm512_and_si512(impl6, impl14);
                __m512i res_i3_715 = _mm512_and_si512(impl7, impl15);

                // Update primes - Stage 1 using AVX512
                __m512i p1_0  = _mm512_andnot_si512(res_i_01,   primes0);
                __m512i p1_1  = _mm512_andnot_si512(res_i_01,   primes1);
                __m512i p1_2  = _mm512_andnot_si512(res_i_23,   primes2);
                __m512i p1_3  = _mm512_andnot_si512(res_i_23,   primes3);
                __m512i p1_4  = _mm512_andnot_si512(res_i_45,   primes4);
                __m512i p1_5  = _mm512_andnot_si512(res_i_45,   primes5);
                __m512i p1_6  = _mm512_andnot_si512(res_i_67,   primes6);
                __m512i p1_7  = _mm512_andnot_si512(res_i_67,   primes7);
                __m512i p1_8  = _mm512_andnot_si512(res_i_89,   primes8);
                __m512i p1_9  = _mm512_andnot_si512(res_i_89,   primes9);
                __m512i p1_10 = _mm512_andnot_si512(res_i_1011, primes10);
                __m512i p1_11 = _mm512_andnot_si512(res_i_1011, primes11);
                __m512i p1_12 = _mm512_andnot_si512(res_i_1213, primes12);
                __m512i p1_13 = _mm512_andnot_si512(res_i_1213, primes13);
                __m512i p1_14 = _mm512_andnot_si512(res_i_1415, primes14);
                __m512i p1_15 = _mm512_andnot_si512(res_i_1415, primes15);

                // Update primes - Stage 2 using AVX512
                __m512i p2_0  = _mm512_andnot_si512(res_i1_02,   p1_0);
                __m512i p2_1  = _mm512_andnot_si512(res_i1_13,   p1_1);
                __m512i p2_2  = _mm512_andnot_si512(res_i1_02,   p1_2);
                __m512i p2_3  = _mm512_andnot_si512(res_i1_13,   p1_3);
                __m512i p2_4  = _mm512_andnot_si512(res_i1_46,   p1_4);
                __m512i p2_5  = _mm512_andnot_si512(res_i1_57,   p1_5);
                __m512i p2_6  = _mm512_andnot_si512(res_i1_46,   p1_6);
                __m512i p2_7  = _mm512_andnot_si512(res_i1_57,   p1_7);
                __m512i p2_8  = _mm512_andnot_si512(res_i1_810,  p1_8);
                __m512i p2_9  = _mm512_andnot_si512(res_i1_911,  p1_9);
                __m512i p2_10 = _mm512_andnot_si512(res_i1_810,  p1_10);
                __m512i p2_11 = _mm512_andnot_si512(res_i1_911,  p1_11);
                __m512i p2_12 = _mm512_andnot_si512(res_i1_1214, p1_12);
                __m512i p2_13 = _mm512_andnot_si512(res_i1_1315, p1_13);
                __m512i p2_14 = _mm512_andnot_si512(res_i1_1214, p1_14);
                __m512i p2_15 = _mm512_andnot_si512(res_i1_1315, p1_15);

                // Update primes - Stage 3 using AVX512
                __m512i p3_0  = _mm512_andnot_si512(res_i2_04,   p2_0);
                __m512i p3_1  = _mm512_andnot_si512(res_i2_15,   p2_1);
                __m512i p3_2  = _mm512_andnot_si512(res_i2_26,   p2_2);
                __m512i p3_3  = _mm512_andnot_si512(res_i2_37,   p2_3);
                __m512i p3_4  = _mm512_andnot_si512(res_i2_04,   p2_4);
                __m512i p3_5  = _mm512_andnot_si512(res_i2_15,   p2_5);
                __m512i p3_6  = _mm512_andnot_si512(res_i2_26,   p2_6);
                __m512i p3_7  = _mm512_andnot_si512(res_i2_37,   p2_7);
                __m512i p3_8  = _mm512_andnot_si512(res_i2_812,  p2_8);
                __m512i p3_9  = _mm512_andnot_si512(res_i2_913,  p2_9);
                __m512i p3_10 = _mm512_andnot_si512(res_i2_1014, p2_10);
                __m512i p3_11 = _mm512_andnot_si512(res_i2_1115, p2_11);
                __m512i p3_12 = _mm512_andnot_si512(res_i2_812,  p2_12);
                __m512i p3_13 = _mm512_andnot_si512(res_i2_913,  p2_13);
                __m512i p3_14 = _mm512_andnot_si512(res_i2_1014, p2_14);
                __m512i p3_15 = _mm512_andnot_si512(res_i2_1115, p2_15);

                // Update primes - Stage 4 (final) using AVX512
                __m512i p4_0  = _mm512_andnot_si512(res_i3_08,  p3_0);
                __m512i p4_1  = _mm512_andnot_si512(res_i3_19,  p3_1);
                __m512i p4_2  = _mm512_andnot_si512(res_i3_210, p3_2);
                __m512i p4_3  = _mm512_andnot_si512(res_i3_311, p3_3);
                __m512i p4_4  = _mm512_andnot_si512(res_i3_412, p3_4);
                __m512i p4_5  = _mm512_andnot_si512(res_i3_513, p3_5);
                __m512i p4_6  = _mm512_andnot_si512(res_i3_614, p3_6);
                __m512i p4_7  = _mm512_andnot_si512(res_i3_715, p3_7);
                __m512i p4_8  = _mm512_andnot_si512(res_i3_08,  p3_8);
                __m512i p4_9  = _mm512_andnot_si512(res_i3_19,  p3_9);
                __m512i p4_10 = _mm512_andnot_si512(res_i3_210, p3_10);
                __m512i p4_11 = _mm512_andnot_si512(res_i3_311, p3_11);
                __m512i p4_12 = _mm512_andnot_si512(res_i3_412, p3_12);
                __m512i p4_13 = _mm512_andnot_si512(res_i3_513, p3_13);
                __m512i p4_14 = _mm512_andnot_si512(res_i3_614, p3_14);
                __m512i p4_15 = _mm512_andnot_si512(res_i3_715, p3_15);

                // Store final prime values using AVX512
                _mm512_store_si512((__m512i *)(primes_index0 +  0 * block_len_b), p4_0);
                _mm512_store_si512((__m512i *)(primes_index0 +  1 * block_len_b), p4_1);
                _mm512_store_si512((__m512i *)(primes_index0 +  2 * block_len_b), p4_2);
                _mm512_store_si512((__m512i *)(primes_index0 +  3 * block_len_b), p4_3);
                _mm512_store_si512((__m512i *)(primes_index0 +  4 * block_len_b), p4_4);
                _mm512_store_si512((__m512i *)(primes_index0 +  5 * block_len_b), p4_5);
                _mm512_store_si512((__m512i *)(primes_index0 +  6 * block_len_b), p4_6);
                _mm512_store_si512((__m512i *)(primes_index0 +  7 * block_len_b), p4_7);
                _mm512_store_si512((__m512i *)(primes_index0 +  8 * block_len_b), p4_8);
                _mm512_store_si512((__m512i *)(primes_index0 +  9 * block_len_b), p4_9);
                _mm512_store_si512((__m512i *)(primes_index0 + 10 * block_len_b), p4_10);
                _mm512_store_si512((__m512i *)(primes_index0 + 11 * block_len_b), p4_11);
                _mm512_store_si512((__m512i *)(primes_index0 + 12 * block_len_b), p4_12);
                _mm512_store_si512((__m512i *)(primes_index0 + 13 * block_len_b), p4_13);
                _mm512_store_si512((__m512i *)(primes_index0 + 14 * block_len_b), p4_14);
                _mm512_store_si512((__m512i *)(primes_index0 + 15 * block_len_b), p4_15);

                // Store new implicants if past the first difference
                if (i >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i - first_difference) << (num_bits - 4));
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i_01);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i_23);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i_45);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i_67);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i_89);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i_1011);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i_1213);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i_1415);
                }
                if (i + 1 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 1 - first_difference) << (num_bits - 4));
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i1_02);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i1_13);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i1_46);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i1_57);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i1_810);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i1_911);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i1_1214);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i1_1315);
                }
                if (i + 2 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 2 - first_difference) << (num_bits - 4));
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i2_04);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i2_15);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i2_26);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i2_37);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i2_812);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i2_913);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i2_1014);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i2_1115);
                }
                if (i + 3 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 3 - first_difference) << (num_bits - 4));
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i3_08);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i3_19);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i3_210);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i3_311);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i3_412);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i3_513);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i3_614);
                    _mm512_store_si512((__m512i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i3_715);
                }

                impl_index0 += 64;
                primes_index0 += 64;
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_AVX512 >= 2
    for (; i+2 < num_bits; i += 3) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 4) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 64) { // Process 64 bytes at a time with AVX512
                __m512i impl0 = _mm512_load_si512((__m512i *)(impl_index0));
                __m512i impl1 = _mm512_load_si512((__m512i *)(impl_index0 + block_len_b));
                __m512i impl2 = _mm512_load_si512((__m512i *)(impl_index0 + 2*block_len_b));
                __m512i impl3 = _mm512_load_si512((__m512i *)(impl_index0 + 3*block_len_b));
                __m512i impl4 = _mm512_load_si512((__m512i *)(impl_index0 + 4*block_len_b));
                __m512i impl5 = _mm512_load_si512((__m512i *)(impl_index0 + 5*block_len_b));
                __m512i impl6 = _mm512_load_si512((__m512i *)(impl_index0 + 6*block_len_b));
                __m512i impl7 = _mm512_load_si512((__m512i *)(impl_index0 + 7*block_len_b));

                __m512i primes0 = _mm512_load_si512((__m512i *)(primes_index0));
                __m512i primes1 = _mm512_load_si512((__m512i *)(primes_index0 + block_len_b));
                __m512i primes2 = _mm512_load_si512((__m512i *)(primes_index0 + 2*block_len_b));
                __m512i primes3 = _mm512_load_si512((__m512i *)(primes_index0 + 3*block_len_b));
                __m512i primes4 = _mm512_load_si512((__m512i *)(primes_index0 + 4*block_len_b));
                __m512i primes5 = _mm512_load_si512((__m512i *)(primes_index0 + 5*block_len_b));
                __m512i primes6 = _mm512_load_si512((__m512i *)(primes_index0 + 6*block_len_b));
                __m512i primes7 = _mm512_load_si512((__m512i *)(primes_index0 + 7*block_len_b));

                __m512i res01 = _mm512_and_si512(impl0, impl1);
                __m512i res23 = _mm512_and_si512(impl2, impl3);
                __m512i res45 = _mm512_and_si512(impl4, impl5);
                __m512i res67 = _mm512_and_si512(impl6, impl7);
                __m512i res02 = _mm512_and_si512(impl0, impl2);
                __m512i res13 = _mm512_and_si512(impl1, impl3);
                __m512i res46 = _mm512_and_si512(impl4, impl6);
                __m512i res57 = _mm512_and_si512(impl5, impl7);
                __m512i res04 = _mm512_and_si512(impl0, impl4);
                __m512i res15 = _mm512_and_si512(impl1, impl5);
                __m512i res26 = _mm512_and_si512(impl2, impl6);
                __m512i res37 = _mm512_and_si512(impl3, impl7);

                __m512i primes0_ = _mm512_andnot_si512(res01, primes0);
                __m512i primes1_ = _mm512_andnot_si512(res01, primes1);
                __m512i primes2_ = _mm512_andnot_si512(res23, primes2);
                __m512i primes3_ = _mm512_andnot_si512(res23, primes3);
                __m512i primes4_ = _mm512_andnot_si512(res45, primes4);
                __m512i primes5_ = _mm512_andnot_si512(res45, primes5);
                __m512i primes6_ = _mm512_andnot_si512(res67, primes6);
                __m512i primes7_ = _mm512_andnot_si512(res67, primes7);

                __m512i primes0__ = _mm512_andnot_si512(res02, primes0_);
                __m512i primes1__ = _mm512_andnot_si512(res13, primes1_);
                __m512i primes2__ = _mm512_andnot_si512(res02, primes2_);
                __m512i primes3__ = _mm512_andnot_si512(res13, primes3_);
                __m512i primes4__ = _mm512_andnot_si512(res46, primes4_);
                __m512i primes5__ = _mm512_andnot_si512(res57, primes5_);
                __m512i primes6__ = _mm512_andnot_si512(res46, primes6_);
                __m512i primes7__ = _mm512_andnot_si512(res57, primes7_);

                __m512i primes0___ = _mm512_andnot_si512(res04, primes0__);
                __m512i primes1___ = _mm512_andnot_si512(res15, primes1__);
                __m512i primes2___ = _mm512_andnot_si512(res26, primes2__);
                __m512i primes3___ = _mm512_andnot_si512(res37, primes3__);
                __m512i primes4___ = _mm512_andnot_si512(res04, primes4__);
                __m512i primes5___ = _mm512_andnot_si512(res15, primes5__);
                __m512i primes6___ = _mm512_andnot_si512(res26, primes6__);
                __m512i primes7___ = _mm512_andnot_si512(res37, primes7__);

                _mm512_store_si512((__m512i *)(primes_index0), primes0___);
                _mm512_store_si512((__m512i *)(primes_index0 + block_len_b), primes1___);
                _mm512_store_si512((__m512i *)(primes_index0 + 2*block_len_b), primes2___);
                _mm512_store_si512((__m512i *)(primes_index0 + 3*block_len_b), primes3___);
                _mm512_store_si512((__m512i *)(primes_index0 + 4*block_len_b), primes4___);
                _mm512_store_si512((__m512i *)(primes_index0 + 5*block_len_b), primes5___);
                _mm512_store_si512((__m512i *)(primes_index0 + 6*block_len_b), primes6___);
                _mm512_store_si512((__m512i *)(primes_index0 + 7*block_len_b), primes7___);

                if (i >= first_difference) {
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res01);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res23);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res45);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res67);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res02);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res13);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res46);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res57);
                }
                if (i+2 >= first_difference) {
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res04);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res15);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res26);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res37);
                }
                impl_index0 += 64;
                primes_index0 += 64;
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE_AVX512 >= 1
    for (; i+1 < num_bits; i += 2) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 2) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 64) { // Process 64 bytes at a time with AVX512
                __m512i impl0 = _mm512_load_si512((__m512i *)(impl_index0));
                __m512i impl1 = _mm512_load_si512((__m512i *)(impl_index0 + block_len_b));
                __m512i impl2 = _mm512_load_si512((__m512i *)(impl_index0 + 2*block_len_b));
                __m512i impl3 = _mm512_load_si512((__m512i *)(impl_index0 + 3*block_len_b));

                __m512i primes0 = _mm512_load_si512((__m512i *)(primes_index0));
                __m512i primes1 = _mm512_load_si512((__m512i *)(primes_index0 + block_len_b));
                __m512i primes2 = _mm512_load_si512((__m512i *)(primes_index0 + 2*block_len_b));
                __m512i primes3 = _mm512_load_si512((__m512i *)(primes_index0 + 3*block_len_b));

                __m512i res01 = _mm512_and_si512(impl0, impl1);
                __m512i res23 = _mm512_and_si512(impl2, impl3);
                __m512i res02 = _mm512_and_si512(impl0, impl2);
                __m512i res13 = _mm512_and_si512(impl1, impl3);

                __m512i primes0_ = _mm512_andnot_si512(res01, primes0);
                __m512i primes1_ = _mm512_andnot_si512(res01, primes1);
                __m512i primes2_ = _mm512_andnot_si512(res23, primes2);
                __m512i primes3_ = _mm512_andnot_si512(res23, primes3);

                __m512i primes0__ = _mm512_andnot_si512(res02, primes0_);
                __m512i primes1__ = _mm512_andnot_si512(res13, primes1_);
                __m512i primes2__ = _mm512_andnot_si512(res02, primes2_);
                __m512i primes3__ = _mm512_andnot_si512(res13, primes3_);

                _mm512_store_si512((__m512i *)(primes_index0), primes0__);
                _mm512_store_si512((__m512i *)(primes_index0 + block_len_b), primes1__);
                _mm512_store_si512((__m512i *)(primes_index0 + 2*block_len_b), primes2__);
                _mm512_store_si512((__m512i *)(primes_index0 + 3*block_len_b), primes3__);

                if (i >= first_difference) {
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res01);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+1) * block_len_b+ k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res23);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res02);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx), res13);
                }
                impl_index0 += 64;
                primes_index0 += 64;
            }
        }
    }
#endif


    for (; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            // Process in 512-bit chunks
            for (int k = 0; k < block_len; k += 512) {
                __m512i impl1 = _mm512_load_si512((__m512i *)(implicants.bits + idx1 / 8));
                __m512i impl2 = _mm512_load_si512((__m512i *)(implicants.bits + idx2 / 8));
                __m512i primes1 = _mm512_load_si512((__m512i *)(primes.bits + idx1 / 8));
                __m512i primes2 = _mm512_load_si512((__m512i *)(primes.bits + idx2 / 8));

                // Using AVX512 intrinsics
                __m512i res = _mm512_and_si512(impl1, impl2);
                __m512i primes1_ = _mm512_andnot_si512(res, primes1);
                __m512i primes2_ = _mm512_andnot_si512(res, primes2);

                _mm512_store_si512((__m512i *)(primes.bits + idx1 / 8), primes1_);
                _mm512_store_si512((__m512i *)(primes.bits + idx2 / 8), primes2_);

                if (i >= first_difference) {
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm512_store_si512((__m512i *)(implicants.bits + o_idx / 8), res);
                }

                idx1 += 512; // Increment by 512 bits
                idx2 += 512; // Increment by 512 bits
            }
        }
    }
}
