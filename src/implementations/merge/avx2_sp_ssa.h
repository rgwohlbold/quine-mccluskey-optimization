#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>

#include "../../bitmap.h"
#include "pext_sp.h"

static void merge_avx2_sp_small_n_ssa(bitmap implicants, bitmap primes, size_t input_index, size_t output_index,
                                      int num_bits, int first_difference) {
    size_t o_idx;
    switch (num_bits) {
        case 0:
            return;
        case 1:
            merge_bits_sp1(implicants, primes, input_index, output_index, first_difference);
            return;
        case 2:
            merge_bits_sp2(implicants, primes, input_index, output_index, first_difference);
            return;
        case 3:
            merge_bits_sp3(implicants, primes, input_index, output_index, first_difference);
            return;
        case 4:
            merge_bits_sp4(implicants, primes, input_index, output_index, first_difference);
            return;
        case 5:
            merge_bits_sp5(implicants, primes, input_index, output_index, first_difference);
            return;
    }

    o_idx = output_index;
    for (int i = 0; i < 6; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 32 / block_len) {
            size_t idx1 = input_index + 2 * block * block_len;

            uint64_t *input_ptr = (uint64_t *)implicants.bits;
            uint32_t *output_ptr = (uint32_t *)implicants.bits;
            uint64_t *primes_ptr = (uint64_t *)primes.bits;
            for (int k = 0; k < block_len; k += 64) {
                uint64_t impl1 = input_ptr[idx1 / 64];
                uint64_t prime;
                if (block_len == 1) {
                    prime = impl1;
                } else {
                    prime = primes_ptr[idx1 / 64];
                }

                uint64_t impl2 = impl1 >> block_len;
                uint64_t aggregated = impl1 & impl2;

                uint64_t mask;
                if (block_len == 1) {
                    mask = 0b0101010101010101010101010101010101010101010101010101010101010101;
                } else if (block_len == 2) {
                    mask = 0b0011001100110011001100110011001100110011001100110011001100110011;
                } else if (block_len == 4) {
                    mask = 0x0F0F0F0F0F0F0F0F;
                } else if (block_len == 8) {
                    mask = 0x00FF00FF00FF00FF;
                } else if (block_len == 16) {
                    mask = 0x0000FFFF0000FFFF;
                } else {  // block_len == 32
                    mask = 0x00000000FFFFFFFF;
                }
                uint64_t result = _pext_u64(aggregated, mask);
                uint64_t initial_result = aggregated & mask;

                uint64_t prime2 = prime & ~(initial_result | (initial_result << block_len));

                primes_ptr[idx1 / 64] = prime2;
                if (i >= first_difference) {
                    output_ptr[o_idx / 32] = (uint32_t)result;
                    o_idx += 32;
                }
                idx1 += 64;
            }
        }
    }

    for (int i = 6; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            for (int k = 0; k < block_len; k += 64) {
                uint64_t *implicant_ptr = (uint64_t *)implicants.bits;
                uint64_t *primes_ptr = (uint64_t *)primes.bits;

                uint64_t impl1 = implicant_ptr[idx1 / 64];
                uint64_t impl2 = implicant_ptr[idx2 / 64];
                uint64_t prime1 = primes_ptr[idx1 / 64];
                uint64_t prime2 = primes_ptr[idx2 / 64];
                uint64_t res = impl1 & impl2;
                uint64_t prime1_ = prime1 & ~res;
                uint64_t prime2_ = prime2 & ~res;

                primes_ptr[idx1 / 64] = prime1_;
                primes_ptr[idx2 / 64] = prime2_;
                if (i >= first_difference) {
                    implicant_ptr[o_idx / 64] = res;
                    o_idx += 64;
                }
                idx1 += 64;
                idx2 += 64;
            }
        }
    }
}

static inline void merge_avx2_sp_single_register_ssa_1(__m256i impl1, __m256i primes1, __m128i *result,
                                                       __m256i *primes_result) {
    const int block_len = 1;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation step
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // block_len == 1
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi8(0b01010101));
    __m256i initial_result = masked0;
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
    __m256i shifted3 = _mm256_srli_epi64(masked3, 8);

    // block_len <= 16
    __m256i combined3 = _mm256_or_si256(masked3, shifted3);
    __m256i masked4 = _mm256_and_si256(combined3, _mm256_set1_epi32(0x0000FFFF));
    __m256i shifted4 = _mm256_srli_epi64(masked4, 16);

    // block_len <= 32
    __m256i combined4 = _mm256_or_si256(masked4, shifted4);
    __m256i masked5 = _mm256_and_si256(combined4, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i shifted5 = _mm256_srli_si256(masked5, 4);  // 4 bytes = 32 bits

    __m256i combined5 = _mm256_or_si256(masked5, shifted5);
    __m256i aggregated_final =
        _mm256_and_si256(combined5, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_2(__m256i impl1, __m256i primes1, __m128i *result,
                                                       __m256i *primes_result) {
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
    __m256i shifted2 = _mm256_srli_epi64(masked2, 8);

    // Block aggregation for length <= 16
    __m256i combined2 = _mm256_or_si256(masked2, shifted2);
    __m256i masked3 = _mm256_and_si256(combined2, _mm256_set1_epi32(0x0000FFFF));
    __m256i shifted3 = _mm256_srli_epi64(masked3, 16);

    // Block aggregation for length <= 32
    __m256i combined3 = _mm256_or_si256(masked3, shifted3);
    __m256i masked4 = _mm256_and_si256(combined3, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i shifted4 = _mm256_srli_si256(masked4, 4);  // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined4 = _mm256_or_si256(masked4, shifted4);
    __m256i aggregated_final =
        _mm256_and_si256(combined4, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_4(__m256i impl1, __m256i primes1, __m128i *result,
                                                       __m256i *primes_result) {
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
    __m256i shifted1 = _mm256_srli_epi64(masked1, 8);

    // Block aggregation for length <= 16
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i masked2 = _mm256_and_si256(combined1, _mm256_set1_epi32(0x0000FFFF));
    __m256i shifted2 = _mm256_srli_epi64(masked2, 16);

    // Block aggregation for length <= 32
    __m256i combined2 = _mm256_or_si256(masked2, shifted2);
    __m256i masked3 = _mm256_and_si256(combined2, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i shifted3 = _mm256_srli_si256(masked3, 4);  // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined3 = _mm256_or_si256(masked3, shifted3);
    __m256i aggregated_final =
        _mm256_and_si256(combined3, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_8(__m256i impl1, __m256i primes1, __m128i *result,
                                                       __m256i *primes_result) {
    const int block_len = 8;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 8
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi16(0x00FF));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_epi64(masked0, 8);

    // Block aggregation for length <= 16
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i masked1 = _mm256_and_si256(combined0, _mm256_set1_epi32(0x0000FFFF));
    __m256i shifted1 = _mm256_srli_epi64(masked1, 16);

    // Block aggregation for length <= 32
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i masked2 = _mm256_and_si256(combined1, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i shifted2 = _mm256_srli_si256(masked2, 4);  // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined2 = _mm256_or_si256(masked2, shifted2);
    __m256i aggregated_final =
        _mm256_and_si256(combined2, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_16(__m256i impl1, __m256i primes1, __m128i *result,
                                                        __m256i *primes_result) {
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
    __m256i shifted1 = _mm256_srli_si256(masked1, 4);  // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined1 = _mm256_or_si256(masked1, shifted1);
    __m256i aggregated_final =
        _mm256_and_si256(combined1, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_32(__m256i impl1, __m256i primes1, __m128i *result,
                                                        __m256i *primes_result) {
    const int block_len = 32;

    // Block_len <= 64: shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply mask for block_len == 32
    __m256i masked0 = _mm256_and_si256(aggregated0, _mm256_set1_epi64x(0x00000000FFFFFFFF));
    __m256i initial_result = masked0;
    __m256i shifted0 = _mm256_srli_si256(masked0, 4);  // 4 bytes = 32 bits

    // Final aggregation
    __m256i combined0 = _mm256_or_si256(masked0, shifted0);
    __m256i aggregated_final =
        _mm256_and_si256(combined0, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // Move 64-bit value to ensure result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated_final, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // Shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_ssa_64(__m256i impl1, __m256i primes1, __m128i *result,
                                                        __m256i *primes_result) {
    // Shift by 64 bits (8 bytes)
    __m256i impl2 = _mm256_srli_si256(impl1, 8);

    // Initial aggregation
    __m256i aggregated0 = _mm256_and_si256(impl1, impl2);

    // Apply 64-bit mask
    __m256i masked0 =
        _mm256_and_si256(aggregated0, _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
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

static inline void merge_avx2_sp_single_register_ssa_128(__m256i impl1, __m256i primes1, __m128i *result,
                                                         __m256i *primes_result) {
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

static inline void merge_avx2_sp_single_register_ssa(int bit_difference, __m256i impl1, __m256i primes1,
                                                     __m128i *result, __m256i *primes_result) {
    int block_len = 1 << bit_difference;

    switch (block_len) {
        case 1:
            merge_avx2_sp_single_register_ssa_1(impl1, primes1, result, primes_result);
            break;
        case 2:
            merge_avx2_sp_single_register_ssa_2(impl1, primes1, result, primes_result);
            break;
        case 4:
            merge_avx2_sp_single_register_ssa_4(impl1, primes1, result, primes_result);
            break;
        case 8:
            merge_avx2_sp_single_register_ssa_8(impl1, primes1, result, primes_result);
            break;
        case 16:
            merge_avx2_sp_single_register_ssa_16(impl1, primes1, result, primes_result);
            break;
        case 32:
            merge_avx2_sp_single_register_ssa_32(impl1, primes1, result, primes_result);
            break;
        case 64:
            merge_avx2_sp_single_register_ssa_64(impl1, primes1, result, primes_result);
            break;
        case 128:
            merge_avx2_sp_single_register_ssa_128(impl1, primes1, result, primes_result);
            break;
        default:
            break;
    }
}

static void merge_avx2_sp_ssa(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits,
                              int first_difference) {
    if (num_bits <= 7) {
        merge_pext_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    for (int i = 0; i < 8; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 128 / block_len) {
            size_t idx1 = input_index + 2 * block * block_len;

            __m256i impl1 = _mm256_load_si256((__m256i *)(implicants.bits + idx1 / 8));
            __m256i primes1;
            // assume we always run block_len=1 first and primes[idx1/8] is uninitialized at this point.
            // in that case, assume all implicants are prime
            if (block_len == 1) {
                primes1 = impl1;
            } else {
                primes1 = _mm256_load_si256((__m256i *)(primes.bits + idx1 / 8));
            }
            __m128i impl_result = _mm_undefined_si128();       // prevent uninitialized warnings
            __m256i primes_result = _mm256_undefined_si256();  // prevent uninitialized warnings;
            merge_avx2_sp_single_register_ssa(i, impl1, primes1, &impl_result, &primes_result);
            _mm256_store_si256((__m256i *)(primes.bits + idx1 / 8), primes_result);
            if (i >= first_difference) {
                _mm_store_si128((__m128i *)(implicants.bits + o_idx / 8), impl_result);
                o_idx += 128;
            }
            idx1 += 256;
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
