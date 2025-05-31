#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>
#include "../../bitmap.h"
#include "bits_sp.h"
#include "../../debug.h"

static void merge_avx2_sp_small_n_aleksa(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits == 0) return;
    
    if (num_bits == 1) {
        merge_bits_sp1(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 2) {
        merge_bits_sp2(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 3) {
        merge_bits_sp3(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 4) {
        merge_bits_sp4(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 5) {
        merge_bits_sp5(implicants, primes, input_index, output_index, first_difference);
        return;
    }
    size_t o_idx = output_index;

    // LOG_INFO("Entered small_n_aleksa with num_bits=%d, first_difference=%d", num_bits, first_difference);

    for (int i = 0; i < 6; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // LOG_INFO("i: %d, block_len: %d, num_blocks: %d", i, block_len, num_blocks);

        for (int block = 0; block < num_blocks; block += 32 / block_len) {
            size_t idx1 = input_index + 2 * block * block_len;

            uint64_t *input_ptr = (uint64_t *) implicants.bits;
            uint32_t *output_ptr = (uint32_t *) implicants.bits;
            uint64_t *primes_ptr = (uint64_t *) primes.bits;
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
                } else { // block_len == 32
                    mask = 0x00000000FFFFFFFF;
                }
                uint64_t result = _pext_u64(aggregated, mask);
                uint64_t initial_result = aggregated & mask;

                uint64_t prime2 = prime & ~(initial_result | (initial_result << block_len));

                primes_ptr[idx1 / 64] = prime2;
                if (i >= first_difference) {
                    output_ptr[o_idx / 32] = (uint32_t) result;
                    o_idx += 32;
                }
                idx1 += 64;
            }
        }

    }


    for (int i = 6; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        // LOG_INFO("i: %d, block_len: %d, num_blocks: %d", i, block_len, num_blocks);

        // implicants do not fit into one register, and we use the largest register size
        for (int block = 0; block < num_blocks; block++) {
            size_t idx1 = input_index + 2 * block * block_len;
            size_t idx2 = input_index + 2 * block * block_len + block_len;

            for (int k = 0; k < block_len; k += 64) {
                uint64_t *implicant_ptr = (uint64_t*) implicants.bits;
                uint64_t *primes_ptr = (uint64_t*) primes.bits;

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


static inline void merge_avx2_sp_single_register_aleksa_1(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {

    const int block_len = 1;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_epi64(impl1, block_len);

    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);

    // block_len == 1
    aggregated = _mm256_and_si256(aggregated, _mm256_set1_epi8(0b01010101));
    initial_result = aggregated;
    shifted = _mm256_srli_epi64(aggregated, 1);

    // block_len <= 2
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00110011));
    shifted = _mm256_srli_epi64(aggregated, 2);

    // block_len <= 4
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
    shifted = _mm256_srli_epi64(aggregated, 4);

    // block_len <= 8
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
    shifted = _mm256_srli_epi64(aggregated, 8);

    // block_len <= 16
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
    shifted = _mm256_srli_epi64(aggregated, 16);

    // block_len <= 32
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
    shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));

    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;

}

static inline void merge_avx2_sp_single_register_aleksa_2(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    const int block_len = 2;
    
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    impl2 = _mm256_srli_epi64(impl1, block_len);


    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00110011));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 2);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
        shifted = _mm256_srli_epi64(aggregated, 4);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
        shifted = _mm256_srli_epi64(aggregated, 8);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        shifted = _mm256_srli_epi64(aggregated, 16);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_aleksa_4(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    const int block_len = 4;
    
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    impl2 = _mm256_srli_epi64(impl1, block_len);


    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 4);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
        shifted = _mm256_srli_epi64(aggregated, 8);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        shifted = _mm256_srli_epi64(aggregated, 16);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_aleksa_8(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    const int block_len = 8;
    
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    impl2 = _mm256_srli_epi64(impl1, block_len);


    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);
        
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 8);
        
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        shifted = _mm256_srli_epi64(aggregated, 16);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_aleksa_16(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    const int block_len = 16;
    
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    impl2 = _mm256_srli_epi64(impl1, block_len);


    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);
        
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 16);

        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_aleksa_32(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    const int block_len = 32;
    
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    impl2 = _mm256_srli_epi64(impl1, block_len);


    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);
        
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        initial_result = aggregated;
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

static inline void merge_avx2_sp_single_register_aleksa_64(__m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    
    // const int block_len = 64;

    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2 = _mm256_srli_si256(impl1, 8); // 8 bytes = 64 bits

    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);

    // TODO: can optimize here (shiftedis always 0)
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    initial_result = aggregated;

    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result = _mm256_slli_si256(initial_result, 8); // 8 bytes = 64 bits

    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;

}


static inline void merge_avx2_sp_single_register_aleksa(int bit_difference, __m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    assert(0 <= bit_difference && bit_difference <= 7);

    int block_len = 1 << bit_difference;
    // __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings

    if (block_len == 128) {
        // set upper half to zero, lower half to upper half of impl1
        __m256i impl1_shuffled = _mm256_permute2x128_si256(impl1, impl1, 0x81);
        __m256i aggregated = _mm256_and_si256(impl1, impl1_shuffled);
        // set upper and lower halves to lower half of aggregated
        __m256i merged = _mm256_permute2x128_si256(aggregated, aggregated, 0x00);
        // use lower half of aggregated
        *result = _mm256_castsi256_si128(aggregated);
        *primes_result = _mm256_andnot_si256(merged, primes1);
        return;
    }


    switch (block_len)
    {
    case 1:
        merge_avx2_sp_single_register_aleksa_1(impl1, primes1, result, primes_result);
        break;
    case 2:
        merge_avx2_sp_single_register_aleksa_2(impl1, primes1, result, primes_result);
        break;
    case 4:
        merge_avx2_sp_single_register_aleksa_4(impl1, primes1, result, primes_result);
        break;
    case 8:
        merge_avx2_sp_single_register_aleksa_8(impl1, primes1, result, primes_result);
        break;
    case 16:
        merge_avx2_sp_single_register_aleksa_16(impl1, primes1, result, primes_result);
        break;
    case 32:
        merge_avx2_sp_single_register_aleksa_32(impl1, primes1, result, primes_result);
        break;
    case 64:
        merge_avx2_sp_single_register_aleksa_64(impl1, primes1, result, primes_result);
        break;
    
    default:
        break;
    }

    // // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    // __m256i impl2;
    // if (block_len == 64) {
    //     // we need to shift across 64-bit boundaries which needs an immediate value
    //     impl2 = _mm256_srli_si256(impl1, 8); // 8 bytes = 64 bits
    // } else {
    //     impl2 = _mm256_srli_epi64(impl1, block_len);
    // }

    // __m256i aggregated = _mm256_and_si256(impl1, impl2);
    // __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    // __m256i shifted = _mm256_set1_epi64x(0);
    // if (block_len == 1) {
    //     aggregated = _mm256_and_si256(aggregated, _mm256_set1_epi8(0b01010101));
    //     initial_result = aggregated;
    //     shifted = _mm256_srli_epi64(aggregated, 1);
    // }
    // if (block_len <= 2) {
    //     aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00110011));
    //     if (block_len == 2) {
    //         initial_result = aggregated;
    //     }
    //     shifted = _mm256_srli_epi64(aggregated, 2);
    // }
    // if (block_len <= 4) {
    //     aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
    //     if (block_len == 4) {
    //         initial_result = aggregated;
    //     }
    //     shifted = _mm256_srli_epi64(aggregated, 4);
    // }
    // if (block_len <= 8) {
    //     aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
    //     if (block_len == 8) {
    //         initial_result = aggregated;
    //     }
    //     shifted = _mm256_srli_epi64(aggregated, 8);
    // }
    // if (block_len <= 16) {
    //     aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
    //     if (block_len == 16) {
    //         initial_result = aggregated;
    //     }
    //     shifted = _mm256_srli_epi64(aggregated, 16);
    // }
    // if (block_len <= 32) {
    //     aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
    //     if (block_len == 32) {
    //         initial_result = aggregated;
    //     }
    //     shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    // }
    // aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    // if (block_len == 64) {
    //     initial_result = aggregated;
    // }
    // // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    // __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    // *result = _mm256_castsi256_si128(result256);

    // // shift initial_result left by block_len
    // __m256i shifted_initial_result;
    // if (block_len == 64) {
    //     shifted_initial_result = _mm256_slli_si256(initial_result, 8); // 8 bytes = 64 bits
    // } else {
    //     shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    // }
    // __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    // __m256i r_ = _mm256_andnot_si256(merged, primes1);
    // *primes_result = r_;
}

static void merge_avx2_sp_aleksa(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits <= 7) {
        merge_avx2_sp_small_n_aleksa(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }

    size_t o_idx = output_index;

    for (int i = 0; i < 8; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        for (int block = 0; block < num_blocks; block += 128 / block_len) {
            size_t idx1 = input_index + 2 * block * block_len;

            __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
            __m256i primes1;
            // assume we always run block_len=1 first and primes[idx1/8] is uninitialized at this point.
            // in that case, assume all implicants are prime
            if (block_len == 1) {
                primes1 = impl1;
            } else {
                primes1 = _mm256_load_si256((__m256i*)(primes.bits + idx1 / 8));
            }
            __m128i impl_result = _mm_set1_epi64x(0); // prevent uninitialized warnings
            __m256i primes_result = _mm256_set1_epi16(0); // prevent uninitialized warnings;
            merge_avx2_sp_single_register_aleksa(i, impl1, primes1, &impl_result, &primes_result);
            _mm256_store_si256((__m256i*)(primes.bits + idx1 / 8), primes_result);
            if (i >= first_difference) {
                _mm_store_si128((__m128i*)(implicants.bits + o_idx / 8), impl_result);
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
                __m256i impl1 = _mm256_load_si256((__m256i*)(implicants.bits + idx1 / 8));
                __m256i impl2 = _mm256_load_si256((__m256i*)(implicants.bits + idx2 / 8));
                __m256i primes1 = _mm256_load_si256((__m256i*)(primes.bits + idx1 / 8));
                __m256i primes2 = _mm256_load_si256((__m256i*)(primes.bits + idx2 / 8));
                __m256i res = _mm256_and_si256(impl1, impl2);
                __m256i primes1_ = _mm256_andnot_si256(res, primes1);
                __m256i primes2_ = _mm256_andnot_si256(res, primes2);
                _mm256_store_si256((__m256i*)(primes.bits + idx1 / 8), primes1_);
                _mm256_store_si256((__m256i*)(primes.bits + idx2 / 8), primes2_);
                if (i >= first_difference) {
                    _mm256_store_si256((__m256i*)(implicants.bits + o_idx / 8), res);
                    o_idx += 256;
                }
                idx1 += 256;
                idx2 += 256;
            }
        }
    }
}
