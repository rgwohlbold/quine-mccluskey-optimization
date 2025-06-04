#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "../../debug.h"
#include "./avx2_sp_ssa.h"
#include "bits_sp.h"
static inline void merge_avx2_sp_single_register_shuffle_1(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 1;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation step
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // block_len == 1
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi8(0b01010101));
    __m256i shifted0 = _mm256_srli_epi64(masked0, 1);

    // block_len <= 2
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i masked1 = _mm256_and_si256(combined0, _mm256_set1_epi8(0b00110011));
    __m256i shifted1 = _mm256_srli_epi64(masked1, 2);

    // block_len <= 4
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i masked2 = _mm256_and_si256(combined1, _mm256_set1_epi8(0b00001111));
    __m256i shifted2 = _mm256_srli_epi64(masked2, 4);

    // block_len <= 8
    __m256i combined2 = _mm256_or_si256(masked2, shifted2);
    __m256i masked3 = _mm256_and_si256(combined2, _mm256_set1_epi16(0x00FF));

    // move all bytes into lower halves of 128-bit lanes
    __m256i shuffle_mask = _mm256_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0,
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0
    );
    __m256i aggregated_final = _mm256_shuffle_epi8(masked3, shuffle_mask);

    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(masked0, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, masked0);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx2_sp_single_register_shuffle_2(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 2;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation step
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 2
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi8(0b00110011));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_epi64(masked0, 2);

    // Block aggregation for length <= 4
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i masked1 = _mm256_and_si256(combined0, _mm256_set1_epi8(0b00001111));
    __m256i shifted1 = _mm256_srli_epi64(masked1, 4);

    // Block aggregation for length <= 8
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i masked2 = _mm256_and_si256(combined1, _mm256_set1_epi16(0x00FF));

    // move all bytes into lower halves of 128-bit lanes
    __m256i shuffle_mask = _mm256_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0,
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0
    );
    __m256i aggregated_final = _mm256_shuffle_epi8(masked2, shuffle_mask);

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_shuffle_4(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 4;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 4
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi8(0b00001111));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_epi64(masked0, 4);

    // Block aggregation for length <= 8
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i masked1 = _mm256_and_si256(combined0, _mm256_set1_epi16(0x00FF));

    // move all bytes into lower halves of 128-bit lanes
    __m256i shuffle_mask = _mm256_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0,
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0
    );
    __m256i aggregated_final = _mm256_shuffle_epi8(masked1, shuffle_mask);

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_shuffle_8(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 8;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 8
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi16(0x00FF));
    __m256i initial_result = masked0;

    // move all bytes into lower halves of 128-bit lanes
    __m256i shuffle_mask = _mm256_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0,
        -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0
    );
    __m256i aggregated_final = _mm256_shuffle_epi8(masked0, shuffle_mask);

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_shuffle_16(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 16;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 16
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi32(0x0000FFFF));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_epi64(masked0, 16);

    // Block aggregation for length <= 32
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i masked1 = _mm256_and_si256(combined0, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i shifted1 = _mm256_srli_si256(masked1, 4); // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i aggregated_final = _mm256_and_si256(combined1, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_shuffle_32(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 32;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 32
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_si256(masked0, 4); // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i aggregated_final = _mm256_and_si256(combined0, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_shuffle_64(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    // Shift by 64 bits (8 bytes)
    __m256i impl2 = _mm256_srli_si256(impl1, 8);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply 64-bit mask
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    __m256i initial_result = masked0;

    // No need for shifted0 as it would be zero

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(masked0, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by 64 bits (8 bytes)
    __m256i shifted_initial_result = _mm256_slli_si256(initial_result, 8);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx2_sp_single_register_shuffle_128(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    // Set upper half to zero, lower half to upper half of impl1
    __m256i impl1_shuffled = _mm256_permute2x128_si256(impl1, impl1, 0x81);

    // And operation between original and shuffled
    __m256i aggregated = _mm256_and_si256(impl1, impl1_shuffled);

    // Set upper and lower halves to lower half of aggregated
    __m256i merged = _mm256_permute2x128_si256(aggregated, aggregated, 0x00);

    // Use lower half of aggregated for result
    *result = _mm256_castsi256_si128(aggregated);

    // Calculate primes result
    __m256i primes_result_value = _mm256_andnot_si256(merged, primes1);
    *primes_result = primes_result_value;
}

static void merge_avx2_sp_shuffle(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                                int first_difference) {
    if (num_bits <= 7) {
        merge_avx2_sp_small_n_ssa(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    int num_registers = (1 << num_bits) / 256;
    if (num_registers < 4) {
        for (int register_index = 0; register_index < num_registers; register_index += 1) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 16 * register_index;

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;
            for (int i = 0; i < 8; i++) {
                __m128i impl_result;
                __m256i primes_result;

                merge_avx2_sp_single_register_ssa(i, impl1, primes1, &impl_result, &primes_result);
                primes1 = primes_result;
                if (i >= first_difference) {
                    _mm_store_si128((__m128i *)(implicants.bits + o_idx1_b), impl_result);
                    o_idx1_b += 16 * num_registers;
                }
            }
            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1);
        }
        if (first_difference <= 8) {
            o_idx += (8 - first_difference) * num_registers * 128;
        }
    } else {
        for (int register_index = 0; register_index < num_registers; register_index += 4) {
            size_t idx1 = input_index + 256 * register_index;
            size_t o_idx1_b = (o_idx >> 3) + 16 * register_index;

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1 = impl1;
            __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 256) / 8));
            __m256i primes2 = impl2;
            __m256i impl3 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 512) / 8));
            __m256i primes3 = impl3;
            __m256i impl4 = _mm256_load_si256((__m256i *)(implicants.bits + (idx1 + 768) / 8));
            __m256i primes4 = impl4;

            __m128i impl1_result, impl2_result, impl3_result, impl4_result;

            merge_avx2_sp_single_register_shuffle_1(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_1(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_1(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_1(impl4, primes4, &impl4_result, &primes4);
            if (0 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_2(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_2(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_2(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_2(impl4, primes4, &impl4_result, &primes4);
            if (1 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_4(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_4(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_4(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_4(impl4, primes4, &impl4_result, &primes4);
            if (2 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_8(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_8(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_8(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_8(impl4, primes4, &impl4_result, &primes4);
            if (3 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_16(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_16(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_16(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_16(impl4, primes4, &impl4_result, &primes4);
            if (4 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_32(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_32(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_32(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_32(impl4, primes4, &impl4_result, &primes4);
            if (5 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_64(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_64(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_64(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_64(impl4, primes4, &impl4_result, &primes4);
            if (6 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            merge_avx2_sp_single_register_shuffle_128(impl1, primes1, &impl1_result, &primes1);
            merge_avx2_sp_single_register_shuffle_128(impl2, primes2, &impl2_result, &primes2);
            merge_avx2_sp_single_register_shuffle_128(impl3, primes3, &impl3_result, &primes3);
            merge_avx2_sp_single_register_shuffle_128(impl4, primes4, &impl4_result, &primes4);
            if (7 >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b)), impl1_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 16)), impl2_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 32)), impl3_result);
                _mm_store_si128((__m128i *)(implicants.bits + (o_idx1_b + 48)), impl4_result);
                o_idx1_b += 16 * num_registers;
            }

            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 256) / 8), primes2);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 512) / 8), primes3);
            _mm256_store_si256((__m256i *)(primes.bits + (idx1 + 768) / 8), primes4);
        }
        if (first_difference <= 8) {
            o_idx += (8 - first_difference) * num_registers * 128;
        }
    }

    for (int i = 8; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            for (int k = 0; k < block_len; k += 256) {
                __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
                __m256i impl2 = _mm256_load_si256((__m256i *)(implicants.bits + idx2 / 8));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes.bits + idx2 / 8));
                __m256i res = _mm256_and_si256(impl1, impl2);
                __m256i primes1_ = _mm256_andnot_si256(res, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res, primes2);
                _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes1_);
                _mm256_store_si256((__m256i *)(primes.bits + idx2 / 8), primes2_);
                if (i >= first_difference) {
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res);
                    o_idx += 256;
                }
                idx1 += 256;
                idx2 += 256;
            }
        }
    }
}
