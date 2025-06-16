#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#ifndef LOG_BLOCK_SIZE
#error "need to define LOG_BLOCK_SIZE"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "./avx2_sp_ssa.h"

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

static void merge_avx2_sp_block(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
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

    int i = 8;
    const size_t input_index_b = input_index / 8;
    const size_t output_index_b = output_index / 8;

#if LOG_BLOCK_SIZE >= 4
    for (; i+3 < num_bits; i += 4) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 8) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 32) {
                // Load 16 implicant chunks
                __m256i impl0  = _mm256_load_si256((__m256i *)(impl_index0 +  0 * block_len_b));
                __m256i impl1  = _mm256_load_si256((__m256i *)(impl_index0 +  1 * block_len_b));
                __m256i impl2  = _mm256_load_si256((__m256i *)(impl_index0 +  2 * block_len_b));
                __m256i impl3  = _mm256_load_si256((__m256i *)(impl_index0 +  3 * block_len_b));
                __m256i impl4  = _mm256_load_si256((__m256i *)(impl_index0 +  4 * block_len_b));
                __m256i impl5  = _mm256_load_si256((__m256i *)(impl_index0 +  5 * block_len_b));
                __m256i impl6  = _mm256_load_si256((__m256i *)(impl_index0 +  6 * block_len_b));
                __m256i impl7  = _mm256_load_si256((__m256i *)(impl_index0 +  7 * block_len_b));
                __m256i impl8  = _mm256_load_si256((__m256i *)(impl_index0 +  8 * block_len_b));
                __m256i impl9  = _mm256_load_si256((__m256i *)(impl_index0 +  9 * block_len_b));
                __m256i impl10 = _mm256_load_si256((__m256i *)(impl_index0 + 10 * block_len_b));
                __m256i impl11 = _mm256_load_si256((__m256i *)(impl_index0 + 11 * block_len_b));
                __m256i impl12 = _mm256_load_si256((__m256i *)(impl_index0 + 12 * block_len_b));
                __m256i impl13 = _mm256_load_si256((__m256i *)(impl_index0 + 13 * block_len_b));
                __m256i impl14 = _mm256_load_si256((__m256i *)(impl_index0 + 14 * block_len_b));
                __m256i impl15 = _mm256_load_si256((__m256i *)(impl_index0 + 15 * block_len_b));

                // Load 16 prime chunks
                __m256i primes0  = _mm256_load_si256((__m256i *)(primes_index0 +  0 * block_len_b));
                __m256i primes1  = _mm256_load_si256((__m256i *)(primes_index0 +  1 * block_len_b));
                __m256i primes2  = _mm256_load_si256((__m256i *)(primes_index0 +  2 * block_len_b));
                __m256i primes3  = _mm256_load_si256((__m256i *)(primes_index0 +  3 * block_len_b));
                __m256i primes4  = _mm256_load_si256((__m256i *)(primes_index0 +  4 * block_len_b));
                __m256i primes5  = _mm256_load_si256((__m256i *)(primes_index0 +  5 * block_len_b));
                __m256i primes6  = _mm256_load_si256((__m256i *)(primes_index0 +  6 * block_len_b));
                __m256i primes7  = _mm256_load_si256((__m256i *)(primes_index0 +  7 * block_len_b));
                __m256i primes8  = _mm256_load_si256((__m256i *)(primes_index0 +  8 * block_len_b));
                __m256i primes9  = _mm256_load_si256((__m256i *)(primes_index0 +  9 * block_len_b));
                __m256i primes10 = _mm256_load_si256((__m256i *)(primes_index0 + 10 * block_len_b));
                __m256i primes11 = _mm256_load_si256((__m256i *)(primes_index0 + 11 * block_len_b));
                __m256i primes12 = _mm256_load_si256((__m256i *)(primes_index0 + 12 * block_len_b));
                __m256i primes13 = _mm256_load_si256((__m256i *)(primes_index0 + 13 * block_len_b));
                __m256i primes14 = _mm256_load_si256((__m256i *)(primes_index0 + 14 * block_len_b));
                __m256i primes15 = _mm256_load_si256((__m256i *)(primes_index0 + 15 * block_len_b));

                // Compute results for all 4 levels
                // Level i (distance 1)
                __m256i res_i_01   = _mm256_and_si256(impl0, impl1);
                __m256i res_i_23   = _mm256_and_si256(impl2, impl3);
                __m256i res_i_45   = _mm256_and_si256(impl4, impl5);
                __m256i res_i_67   = _mm256_and_si256(impl6, impl7);
                __m256i res_i_89   = _mm256_and_si256(impl8, impl9);
                __m256i res_i_1011 = _mm256_and_si256(impl10, impl11);
                __m256i res_i_1213 = _mm256_and_si256(impl12, impl13);
                __m256i res_i_1415 = _mm256_and_si256(impl14, impl15);

                // Level i+1 (distance 2)
                __m256i res_i1_02   = _mm256_and_si256(impl0, impl2);
                __m256i res_i1_13   = _mm256_and_si256(impl1, impl3);
                __m256i res_i1_46   = _mm256_and_si256(impl4, impl6);
                __m256i res_i1_57   = _mm256_and_si256(impl5, impl7);
                __m256i res_i1_810  = _mm256_and_si256(impl8, impl10);
                __m256i res_i1_911  = _mm256_and_si256(impl9, impl11);
                __m256i res_i1_1214 = _mm256_and_si256(impl12, impl14);
                __m256i res_i1_1315 = _mm256_and_si256(impl13, impl15);

                // Level i+2 (distance 4)
                __m256i res_i2_04 = _mm256_and_si256(impl0, impl4);
                __m256i res_i2_15 = _mm256_and_si256(impl1, impl5);
                __m256i res_i2_26 = _mm256_and_si256(impl2, impl6);
                __m256i res_i2_37 = _mm256_and_si256(impl3, impl7);
                __m256i res_i2_812 = _mm256_and_si256(impl8, impl12);
                __m256i res_i2_913 = _mm256_and_si256(impl9, impl13);
                __m256i res_i2_1014 = _mm256_and_si256(impl10, impl14);
                __m256i res_i2_1115 = _mm256_and_si256(impl11, impl15);

                // Level i+3 (distance 8)
                __m256i res_i3_08  = _mm256_and_si256(impl0, impl8);
                __m256i res_i3_19  = _mm256_and_si256(impl1, impl9);
                __m256i res_i3_210 = _mm256_and_si256(impl2, impl10);
                __m256i res_i3_311 = _mm256_and_si256(impl3, impl11);
                __m256i res_i3_412 = _mm256_and_si256(impl4, impl12);
                __m256i res_i3_513 = _mm256_and_si256(impl5, impl13);
                __m256i res_i3_614 = _mm256_and_si256(impl6, impl14);
                __m256i res_i3_715 = _mm256_and_si256(impl7, impl15);

                // Update primes - Stage 1
                __m256i p1_0  = _mm256_andnot_si256(res_i_01,   primes0);
                __m256i p1_1  = _mm256_andnot_si256(res_i_01,   primes1);
                __m256i p1_2  = _mm256_andnot_si256(res_i_23,   primes2);
                __m256i p1_3  = _mm256_andnot_si256(res_i_23,   primes3);
                __m256i p1_4  = _mm256_andnot_si256(res_i_45,   primes4);
                __m256i p1_5  = _mm256_andnot_si256(res_i_45,   primes5);
                __m256i p1_6  = _mm256_andnot_si256(res_i_67,   primes6);
                __m256i p1_7  = _mm256_andnot_si256(res_i_67,   primes7);
                __m256i p1_8  = _mm256_andnot_si256(res_i_89,   primes8);
                __m256i p1_9  = _mm256_andnot_si256(res_i_89,   primes9);
                __m256i p1_10 = _mm256_andnot_si256(res_i_1011, primes10);
                __m256i p1_11 = _mm256_andnot_si256(res_i_1011, primes11);
                __m256i p1_12 = _mm256_andnot_si256(res_i_1213, primes12);
                __m256i p1_13 = _mm256_andnot_si256(res_i_1213, primes13);
                __m256i p1_14 = _mm256_andnot_si256(res_i_1415, primes14);
                __m256i p1_15 = _mm256_andnot_si256(res_i_1415, primes15);

                // Update primes - Stage 2
                __m256i p2_0  = _mm256_andnot_si256(res_i1_02,   p1_0);
                __m256i p2_1  = _mm256_andnot_si256(res_i1_13,   p1_1);
                __m256i p2_2  = _mm256_andnot_si256(res_i1_02,   p1_2);
                __m256i p2_3  = _mm256_andnot_si256(res_i1_13,   p1_3);
                __m256i p2_4  = _mm256_andnot_si256(res_i1_46,   p1_4);
                __m256i p2_5  = _mm256_andnot_si256(res_i1_57,   p1_5);
                __m256i p2_6  = _mm256_andnot_si256(res_i1_46,   p1_6);
                __m256i p2_7  = _mm256_andnot_si256(res_i1_57,   p1_7);
                __m256i p2_8  = _mm256_andnot_si256(res_i1_810,  p1_8);
                __m256i p2_9  = _mm256_andnot_si256(res_i1_911,  p1_9);
                __m256i p2_10 = _mm256_andnot_si256(res_i1_810,  p1_10);
                __m256i p2_11 = _mm256_andnot_si256(res_i1_911,  p1_11);
                __m256i p2_12 = _mm256_andnot_si256(res_i1_1214, p1_12);
                __m256i p2_13 = _mm256_andnot_si256(res_i1_1315, p1_13);
                __m256i p2_14 = _mm256_andnot_si256(res_i1_1214, p1_14);
                __m256i p2_15 = _mm256_andnot_si256(res_i1_1315, p1_15);

                // Update primes - Stage 3
                __m256i p3_0  = _mm256_andnot_si256(res_i2_04, p2_0);
                __m256i p3_1  = _mm256_andnot_si256(res_i2_15, p2_1);
                __m256i p3_2  = _mm256_andnot_si256(res_i2_26, p2_2);
                __m256i p3_3  = _mm256_andnot_si256(res_i2_37, p2_3);
                __m256i p3_4  = _mm256_andnot_si256(res_i2_04, p2_4);
                __m256i p3_5  = _mm256_andnot_si256(res_i2_15, p2_5);
                __m256i p3_6  = _mm256_andnot_si256(res_i2_26, p2_6);
                __m256i p3_7  = _mm256_andnot_si256(res_i2_37, p2_7);
                __m256i p3_8  = _mm256_andnot_si256(res_i2_812, p2_8);
                __m256i p3_9  = _mm256_andnot_si256(res_i2_913, p2_9);
                __m256i p3_10 = _mm256_andnot_si256(res_i2_1014, p2_10);
                __m256i p3_11 = _mm256_andnot_si256(res_i2_1115, p2_11);
                __m256i p3_12 = _mm256_andnot_si256(res_i2_812, p2_12);
                __m256i p3_13 = _mm256_andnot_si256(res_i2_913, p2_13);
                __m256i p3_14 = _mm256_andnot_si256(res_i2_1014, p2_14);
                __m256i p3_15 = _mm256_andnot_si256(res_i2_1115, p2_15);

                // Update primes - Stage 4 (final)
                __m256i p4_0  = _mm256_andnot_si256(res_i3_08,  p3_0);
                __m256i p4_1  = _mm256_andnot_si256(res_i3_19,  p3_1);
                __m256i p4_2  = _mm256_andnot_si256(res_i3_210, p3_2);
                __m256i p4_3  = _mm256_andnot_si256(res_i3_311, p3_3);
                __m256i p4_4  = _mm256_andnot_si256(res_i3_412, p3_4);
                __m256i p4_5  = _mm256_andnot_si256(res_i3_513, p3_5);
                __m256i p4_6  = _mm256_andnot_si256(res_i3_614, p3_6);
                __m256i p4_7  = _mm256_andnot_si256(res_i3_715, p3_7);
                __m256i p4_8  = _mm256_andnot_si256(res_i3_08,  p3_8);
                __m256i p4_9  = _mm256_andnot_si256(res_i3_19,  p3_9);
                __m256i p4_10 = _mm256_andnot_si256(res_i3_210, p3_10);
                __m256i p4_11 = _mm256_andnot_si256(res_i3_311, p3_11);
                __m256i p4_12 = _mm256_andnot_si256(res_i3_412, p3_12);
                __m256i p4_13 = _mm256_andnot_si256(res_i3_513, p3_13);
                __m256i p4_14 = _mm256_andnot_si256(res_i3_614, p3_14);
                __m256i p4_15 = _mm256_andnot_si256(res_i3_715, p3_15);

                // Store final prime values
                _mm256_store_si256((__m256i *)(primes_index0 +  0 * block_len_b), p4_0);
                _mm256_store_si256((__m256i *)(primes_index0 +  1 * block_len_b), p4_1);
                _mm256_store_si256((__m256i *)(primes_index0 +  2 * block_len_b), p4_2);
                _mm256_store_si256((__m256i *)(primes_index0 +  3 * block_len_b), p4_3);
                _mm256_store_si256((__m256i *)(primes_index0 +  4 * block_len_b), p4_4);
                _mm256_store_si256((__m256i *)(primes_index0 +  5 * block_len_b), p4_5);
                _mm256_store_si256((__m256i *)(primes_index0 +  6 * block_len_b), p4_6);
                _mm256_store_si256((__m256i *)(primes_index0 +  7 * block_len_b), p4_7);
                _mm256_store_si256((__m256i *)(primes_index0 +  8 * block_len_b), p4_8);
                _mm256_store_si256((__m256i *)(primes_index0 +  9 * block_len_b), p4_9);
                _mm256_store_si256((__m256i *)(primes_index0 + 10 * block_len_b), p4_10);
                _mm256_store_si256((__m256i *)(primes_index0 + 11 * block_len_b), p4_11);
                _mm256_store_si256((__m256i *)(primes_index0 + 12 * block_len_b), p4_12);
                _mm256_store_si256((__m256i *)(primes_index0 + 13 * block_len_b), p4_13);
                _mm256_store_si256((__m256i *)(primes_index0 + 14 * block_len_b), p4_14);
                _mm256_store_si256((__m256i *)(primes_index0 + 15 * block_len_b), p4_15);

                // Store new implicants if past the first difference
                if (i >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i - first_difference) << (num_bits - 4));
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i_01);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i_23);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i_45);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i_67);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i_89);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i_1011);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i_1213);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i_1415);
                }
                if (i + 1 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 1 - first_difference) << (num_bits - 4));
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i1_02);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i1_13);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i1_46);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i1_57);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i1_810);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i1_911);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i1_1214);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i1_1315);
                }
                if (i + 2 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 2 - first_difference) << (num_bits - 4));
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i2_04);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i2_15);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i2_26);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i2_37);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i2_812);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i2_913);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i2_1014);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i2_1115);
                }
                if (i + 3 >= first_difference) {
                    size_t level_base_idx = output_index_b + ((i + 3 - first_difference) << (num_bits - 4));
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 0) * block_len_b + k), res_i3_08);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 1) * block_len_b + k), res_i3_19);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 2) * block_len_b + k), res_i3_210);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 3) * block_len_b + k), res_i3_311);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 4) * block_len_b + k), res_i3_412);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 5) * block_len_b + k), res_i3_513);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 6) * block_len_b + k), res_i3_614);
                    _mm256_store_si256((__m256i *)(implicants.bits + level_base_idx + (block + 7) * block_len_b + k), res_i3_715);
                }

                impl_index0 += 32;
                primes_index0 += 32;
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE >= 3
    for (; i+2 < num_bits; i += 3) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 4) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 32) {
                __m256i impl0 = _mm256_load_si256((__m256i *)(impl_index0));
                __m256i impl1 = _mm256_load_si256((__m256i *)(impl_index0 + block_len_b));
                __m256i impl2 = _mm256_load_si256((__m256i *)(impl_index0 + 2*block_len_b));
                __m256i impl3 = _mm256_load_si256((__m256i *)(impl_index0 + 3*block_len_b));
                __m256i impl4 = _mm256_load_si256((__m256i *)(impl_index0 + 4*block_len_b));
                __m256i impl5 = _mm256_load_si256((__m256i *)(impl_index0 + 5*block_len_b));
                __m256i impl6 = _mm256_load_si256((__m256i *)(impl_index0 + 6*block_len_b));
                __m256i impl7 = _mm256_load_si256((__m256i *)(impl_index0 + 7*block_len_b));


                __m256i primes0 = _mm256_load_si256((__m256i *)(primes_index0));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes_index0 + block_len_b));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes_index0 + 2*block_len_b));
                __m256i primes3 = _mm256_load_si256((__m256i *)(primes_index0 + 3*block_len_b));
                __m256i primes4 = _mm256_load_si256((__m256i *)(primes_index0 + 4*block_len_b));
                __m256i primes5 = _mm256_load_si256((__m256i *)(primes_index0 + 5*block_len_b));
                __m256i primes6 = _mm256_load_si256((__m256i *)(primes_index0 + 6*block_len_b));
                __m256i primes7 = _mm256_load_si256((__m256i *)(primes_index0 + 7*block_len_b));

                __m256i res01 = _mm256_and_si256(impl0, impl1);
                __m256i res23 = _mm256_and_si256(impl2, impl3);
                __m256i res45 = _mm256_and_si256(impl4, impl5);
                __m256i res67 = _mm256_and_si256(impl6, impl7);
                __m256i res02 = _mm256_and_si256(impl0, impl2);
                __m256i res13 = _mm256_and_si256(impl1, impl3);
                __m256i res46 = _mm256_and_si256(impl4, impl6);
                __m256i res57 = _mm256_and_si256(impl5, impl7);
                __m256i res04 = _mm256_and_si256(impl0, impl4);
                __m256i res15 = _mm256_and_si256(impl1, impl5);
                __m256i res26 = _mm256_and_si256(impl2, impl6);
                __m256i res37 = _mm256_and_si256(impl3, impl7);

                __m256i primes0_ = _mm256_andnot_si256(res01, primes0);
                __m256i primes1_ = _mm256_andnot_si256(res01, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res23, primes2);
                __m256i primes3_ = _mm256_andnot_si256(res23, primes3);
                __m256i primes4_ = _mm256_andnot_si256(res45, primes4);
                __m256i primes5_ = _mm256_andnot_si256(res45, primes5);
                __m256i primes6_ = _mm256_andnot_si256(res67, primes6);
                __m256i primes7_ = _mm256_andnot_si256(res67, primes7);

                __m256i primes0__ = _mm256_andnot_si256(res02, primes0_);
                __m256i primes1__ = _mm256_andnot_si256(res13, primes1_);
                __m256i primes2__ = _mm256_andnot_si256(res02, primes2_);
                __m256i primes3__ = _mm256_andnot_si256(res13, primes3_);
                __m256i primes4__ = _mm256_andnot_si256(res46, primes4_);
                __m256i primes5__ = _mm256_andnot_si256(res57, primes5_);
                __m256i primes6__ = _mm256_andnot_si256(res46, primes6_);
                __m256i primes7__ = _mm256_andnot_si256(res57, primes7_);

                __m256i primes0___ = _mm256_andnot_si256(res04, primes0__);
                __m256i primes1___ = _mm256_andnot_si256(res15, primes1__);
                __m256i primes2___ = _mm256_andnot_si256(res26, primes2__);
                __m256i primes3___ = _mm256_andnot_si256(res37, primes3__);
                __m256i primes4___ = _mm256_andnot_si256(res04, primes4__);
                __m256i primes5___ = _mm256_andnot_si256(res15, primes5__);
                __m256i primes6___ = _mm256_andnot_si256(res26, primes6__);
                __m256i primes7___ = _mm256_andnot_si256(res37, primes7__);

                _mm256_store_si256((__m256i *)(primes_index0), primes0___);
                _mm256_store_si256((__m256i *)(primes_index0 + block_len_b), primes1___);
                _mm256_store_si256((__m256i *)(primes_index0 + 2*block_len_b), primes2___);
                _mm256_store_si256((__m256i *)(primes_index0 + 3*block_len_b), primes3___);
                _mm256_store_si256((__m256i *)(primes_index0 + 4*block_len_b), primes4___);
                _mm256_store_si256((__m256i *)(primes_index0 + 5*block_len_b), primes5___);
                _mm256_store_si256((__m256i *)(primes_index0 + 6*block_len_b), primes6___);
                _mm256_store_si256((__m256i *)(primes_index0 + 7*block_len_b), primes7___);
                if (i >= first_difference) {
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res01);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res23);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res45);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res67);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res02);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res13);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res46);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res57);
                }
                if (i+2 >= first_difference) {
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res04);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res15);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+2) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res26);
                    o_idx = output_index_b + ((i+2 - first_difference) << (num_bits - 4)) + (block+3) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res37);
                }
                impl_index0 += 32;
                primes_index0 += 32;
            }
        }
    }
#endif

#if LOG_BLOCK_SIZE >= 1
    for (; i+1 < num_bits; i += 2) {
        int block_len_b = 1 << (i-3);
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block += 2) {
            uint8_t *impl_index0 = &implicants.bits[input_index_b + 2 * block * block_len_b];
            uint8_t *primes_index0 = &primes.bits[input_index_b + 2 * block * block_len_b];

            for (int k = 0; k < block_len_b; k += 32) {
                __m256i impl0 = _mm256_load_si256((__m256i *)(impl_index0));
                __m256i impl1 = _mm256_load_si256((__m256i *)(impl_index0 + block_len_b));
                __m256i impl2 = _mm256_load_si256((__m256i *)(impl_index0 + 2*block_len_b));
                __m256i impl3 = _mm256_load_si256((__m256i *)(impl_index0 + 3*block_len_b));

                __m256i primes0 = _mm256_load_si256((__m256i *)(primes_index0));
                __m256i primes1 = _mm256_load_si256((__m256i *)(primes_index0 + block_len_b));
                __m256i primes2 = _mm256_load_si256((__m256i *)(primes_index0 + 2*block_len_b));
                __m256i primes3 = _mm256_load_si256((__m256i *)(primes_index0 + 3*block_len_b));

                __m256i res01 = _mm256_and_si256(impl0, impl1);
                __m256i res23 = _mm256_and_si256(impl2, impl3);
                __m256i res02 = _mm256_and_si256(impl0, impl2);
                __m256i res13 = _mm256_and_si256(impl1, impl3);

                __m256i primes0_ = _mm256_andnot_si256(res01, primes0);
                __m256i primes1_ = _mm256_andnot_si256(res01, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res23, primes2);
                __m256i primes3_ = _mm256_andnot_si256(res23, primes3);

                __m256i primes0__ = _mm256_andnot_si256(res02, primes0_);
                __m256i primes1__ = _mm256_andnot_si256(res13, primes1_);
                __m256i primes2__ = _mm256_andnot_si256(res02, primes2_);
                __m256i primes3__ = _mm256_andnot_si256(res13, primes3_);

                _mm256_store_si256((__m256i *)(primes_index0), primes0__);
                _mm256_store_si256((__m256i *)(primes_index0 + block_len_b), primes1__);
                _mm256_store_si256((__m256i *)(primes_index0 + 2*block_len_b), primes2__);
                _mm256_store_si256((__m256i *)(primes_index0 + 3*block_len_b), primes3__);
                if (i >= first_difference) {
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res01);
                    o_idx = output_index_b + ((i - first_difference) << (num_bits - 4)) + (block+1) * block_len_b+ k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res23);
                }
                if (i+1 >= first_difference) {
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + block * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res02);
                    o_idx = output_index_b + ((i+1 - first_difference) << (num_bits - 4)) + (block+1) * block_len_b + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx), res13);
                }
                impl_index0 += 32;
                primes_index0 += 32;
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
                    o_idx = output_index + ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    _mm256_store_si256((__m256i *)(implicants.bits + o_idx / 8), res);
                }
                idx1 += 256;
                idx2 += 256;
            }
        }
    }
}
