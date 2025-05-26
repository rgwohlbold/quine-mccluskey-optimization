#include <immintrin.h>
#ifdef __AVX2__
#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"
#include "../vtune.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../debug.h"
#include <x86intrin.h>

#include "bits.h"

#ifndef __BMI2__
#error "need BMI2 as base case to avx2_single_pass"
#endif

static void merge_implicants_bits1(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int first_difference) {
    bool impl1 = BITMAP_CHECK(implicants, input_index);
    bool impl2 = BITMAP_CHECK(implicants, input_index+1);

    bool res = impl1 && impl2;
    bool prime1_ = impl1 && !impl2;
    bool prime2_ = impl2 && !impl1;
    BITMAP_SET(primes, input_index, prime1_);
    BITMAP_SET(primes, input_index+1, prime2_);
    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res);
    }
}

static void merge_implicants_bits2(bitmap implicants, bitmap prime, size_t input_index, size_t output_index, int first_difference) {
    bool impl0 = BITMAP_CHECK(implicants, input_index);
    bool impl1 = BITMAP_CHECK(implicants, input_index+1);
    bool impl2 = BITMAP_CHECK(implicants, input_index+2);
    bool impl3 = BITMAP_CHECK(implicants, input_index+3);
    bool res0 = impl0 && impl1;
    bool res1 = impl2 && impl3;

    bool res2 = impl0 && impl2;
    bool res3 = impl1 && impl3;

    bool prime0_ = impl0 && !(impl1 || impl2);
    bool prime1_ = impl1 && !(impl0 || impl3);
    bool prime2_ = impl2 && !(impl0 || impl3);
    bool prime3_ = impl3 && !(impl1 || impl2);
    BITMAP_SET(prime, input_index, prime0_);
    BITMAP_SET(prime, input_index+1, prime1_);
    BITMAP_SET(prime, input_index+2, prime2_);
    BITMAP_SET(prime, input_index+3, prime3_);
    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res0);
        BITMAP_SET(implicants, output_index+1, res1);
        BITMAP_SET(implicants, output_index+2, res2);
        BITMAP_SET(implicants, output_index+3, res3);
    } else if (first_difference == 1) {
        BITMAP_SET(implicants, output_index, res2);
        BITMAP_SET(implicants, output_index+1, res3);
    }
}

static void merge_implicants_bits3(
    bitmap implicants,
    bitmap primes,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    // TODO: load everything as byte and perform bitwise operations
    bool impl0 = BITMAP_CHECK(implicants, input_index);
    bool impl1 = BITMAP_CHECK(implicants, input_index+1);
    bool impl2 = BITMAP_CHECK(implicants, input_index+2);
    bool impl3 = BITMAP_CHECK(implicants, input_index+3);
    bool impl4 = BITMAP_CHECK(implicants, input_index+4);
    bool impl5 = BITMAP_CHECK(implicants, input_index+5);
    bool impl6 = BITMAP_CHECK(implicants, input_index+6);
    bool impl7 = BITMAP_CHECK(implicants, input_index+7);
    bool res0 = impl0 && impl1;
    bool res1 = impl2 && impl3;
    bool res2 = impl4 && impl5;
    bool res3 = impl6 && impl7;
    bool res4 = impl0 && impl2;
    bool res5 = impl1 && impl3;
    bool res6 = impl4 && impl6;
    bool res7 = impl5 && impl7;
    bool res8 = impl0 && impl4;
    bool res9 = impl1 && impl5;
    bool res10 = impl2 && impl6;
    bool res11 = impl3 && impl7;
    bool prime0_ = impl0 && !(res0 || res4 || res8);
    bool prime1_ = impl1 && !(res0 || res5 || res9);
    bool prime2_ = impl2 && !(res1 || res4 || res10);
    bool prime3_ = impl3 && !(res1 || res5 || res11);
    bool prime4_ = impl4 && !(res2 || res6 || res8);
    bool prime5_ = impl5 && !(res2 || res7 || res9);
    bool prime6_ = impl6 && !(res3 || res6 || res10);
    bool prime7_ = impl7 && !(res3 || res7 || res11);
    BITMAP_SET(primes, input_index, prime0_);
    BITMAP_SET(primes, input_index+1, prime1_);
    BITMAP_SET(primes, input_index+2, prime2_);
    BITMAP_SET(primes, input_index+3, prime3_);
    BITMAP_SET(primes, input_index+4, prime4_);
    BITMAP_SET(primes, input_index+5, prime5_);
    BITMAP_SET(primes, input_index+6, prime6_);
    BITMAP_SET(primes, input_index+7, prime7_);

    if (first_difference == 0) {
        BITMAP_SET(implicants, output_index, res0);
        BITMAP_SET(implicants, output_index+1, res1);
        BITMAP_SET(implicants, output_index+2, res2);
        BITMAP_SET(implicants, output_index+3, res3);
        BITMAP_SET(implicants, output_index+4, res4);
        BITMAP_SET(implicants, output_index+5, res5);
        BITMAP_SET(implicants, output_index+6, res6);
        BITMAP_SET(implicants, output_index+7, res7);
        BITMAP_SET(implicants, output_index+8, res8);
        BITMAP_SET(implicants, output_index+9, res9);
        BITMAP_SET(implicants, output_index+10, res10);
        BITMAP_SET(implicants, output_index+11, res11);
    } else if (first_difference == 1) {
        BITMAP_SET(implicants, output_index, res4);
        BITMAP_SET(implicants, output_index+1, res5);
        BITMAP_SET(implicants, output_index+2, res6);
        BITMAP_SET(implicants, output_index+3, res7);
        BITMAP_SET(implicants, output_index+4, res8);
        BITMAP_SET(implicants, output_index+5, res9);
        BITMAP_SET(implicants, output_index+6, res10);
        BITMAP_SET(implicants, output_index+7, res11);
    } else if (first_difference == 2) {
        BITMAP_SET(implicants, output_index, res8);
        BITMAP_SET(implicants, output_index+1, res9);
        BITMAP_SET(implicants, output_index+2, res10);
        BITMAP_SET(implicants, output_index+3, res11);
    }
}

static void merge_implicants_bits4(
    bitmap implicants,
    bitmap primes,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    uint16_t *implicant_ptr = (uint16_t *) implicants.bits;
    uint16_t *primes_ptr = (uint16_t *) primes.bits;
    uint8_t *output_ptr = (uint8_t *) implicants.bits;

    uint16_t impl = implicant_ptr[input_index/16];

    // block size 1 (difference 2^0 = 1)
    uint16_t impl10 = impl & (impl >> 1) & 0x5555;
    uint16_t impl10s = impl10 >> 1;
    uint16_t impl11 = (impl10 | impl10s) & 0x3333;
    uint16_t impl11s = impl11 >> 2;
    uint16_t impl12 = (impl11 | impl11s) & 0x0F0F;
    uint16_t impl12s = impl12 >> 4;
    uint16_t impl1res16 = (impl12 | impl12s) & 0x00FF; // Result fits in 8 bits
    uint16_t merged1 = impl10 | (impl10 << 1);

    // block size 2 (difference 2^1 = 2)
    uint16_t impl21 = impl & (impl >> 2) & 0x3333;
    uint16_t impl21s = impl21 >> 2;
    uint16_t impl22 = (impl21 | impl21s) & 0x0F0F;
    uint16_t impl22s = impl22 >> 4;
    uint16_t impl2res16 = (impl22 | impl22s) & 0x00FF;
    uint16_t merged2 = impl21 | (impl21 << 2);

    // block size 4 (difference 2^2 = 4)
    uint16_t impl32 = impl & (impl >> 4) & 0x0F0F;
    uint16_t impl32s = impl32 >> 4;
    uint16_t impl3res16 = (impl32 | impl32s) & 0x00FF;
    uint16_t merged3 = impl32 | (impl32 << 4);

    // block size 8 (difference 2^3 = 8)
    uint16_t impl4res16 = impl & (impl >> 8) & 0x00FF;
    uint16_t merged4 = impl4res16 | (impl4res16 << 8);

    // --- Combine merged markers (uint16_t) ---
    uint16_t res_prime = impl & ~(merged1 | merged2 | merged3 | merged4);
    primes_ptr[input_index/16] = res_prime;

    // --- Write output implicants (uint8_t) ---
    if (first_difference == 0) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl1res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl2res16;
        output_ptr[(output_index/8) + 2] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 3] = (uint8_t)impl4res16;
    } else if (first_difference == 1) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl2res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 2] = (uint8_t)impl4res16;
    } else if (first_difference == 2) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl3res16;
        output_ptr[(output_index/8) + 1] = (uint8_t)impl4res16;
    } else if (first_difference == 3) {
        output_ptr[(output_index/8)    ] = (uint8_t)impl4res16;
    }
}


static void merge_implicants_bits5(
    bitmap implicants,
    bitmap primes,
    size_t input_index,
    size_t output_index,
    int first_difference
) {
    uint32_t *implicant_ptr = (uint32_t *) implicants.bits;
    uint32_t *primes_ptr = (uint32_t *) primes.bits;
    uint16_t *output_ptr = (uint16_t *) implicants.bits;

    uint32_t impl = implicant_ptr[input_index/32];

    // block size 1 (difference 2^0 = 1)
    uint32_t mask0 = 0b01010101010101010101010101010101;
    uint32_t impl10 = impl & (impl >> 1) & mask0;
    uint32_t impl11 = _pext_u32(impl10, mask0);
    uint32_t merged1 = impl10 | (impl10 << 1);

    // block size 2 (difference 2^1 = 2)
    uint32_t mask1 = 0b00110011001100110011001100110011;
    uint32_t impl20 = impl & (impl >> 2) & mask1;
    uint32_t impl21 = _pext_u32(impl20, mask1);
    uint32_t merged2 = impl20 | (impl20 << 2);

    // block size 4 (difference 2^2 = 4)
    uint32_t mask2 = 0x0F0F0F0F;
    uint32_t impl30 = impl & (impl >> 4) & mask2;
    uint32_t impl31 = _pext_u32(impl30, mask2);
    uint32_t merged3 = impl30 | (impl30 << 4);

    // block size 8 (difference 2^3 = 8)
    uint32_t mask3 = 0x00FF00FF;
    uint32_t impl40 = impl & (impl >> 8) & mask3;
    uint32_t impl41 = _pext_u32(impl40, mask3);
    uint32_t merged4 = impl40 | (impl40 << 8);

    // block size 16 (difference 2^4 = 16)
    uint32_t mask4 = 0x0000FFFF;
    uint32_t impl50 = impl & (impl >> 16) & mask4;
    uint32_t impl51 = impl50; // pext is not needed here
    uint32_t merged5 = impl50 | (impl50 << 16);

    uint32_t res_prime = impl & ~(merged1 | merged2 | merged3 | merged4 | merged5);
    primes_ptr[input_index/32] = res_prime;

    if (first_difference == 0) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl11;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl21;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl31;
        output_ptr[(output_index/16) + 3] = (uint16_t)impl41;
        output_ptr[(output_index/16) + 4] = (uint16_t)impl51;
    } else if (first_difference == 1) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl21;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl31;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl41;
        output_ptr[(output_index/16) + 3] = (uint16_t)impl51;
    } else if (first_difference == 2) {
        output_ptr[(output_index/16)    ] = (uint16_t)impl31;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl41;
        output_ptr[(output_index/16) + 2] = (uint16_t)impl51;
    } else if (first_difference == 3) {
        output_ptr[(output_index/16)] = (uint16_t)impl41;
        output_ptr[(output_index/16) + 1] = (uint16_t)impl51;
    } else if (first_difference == 4) {
        output_ptr[output_index/16] = (uint16_t)impl51;
    }
}

static void merge_implicants_avx2_single_pass_small_n(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits == 1) {
        merge_implicants_bits1(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 2) {
        merge_implicants_bits2(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 3) {
        merge_implicants_bits3(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 4) {
        merge_implicants_bits4(implicants, primes, input_index, output_index, first_difference);
        return;
    } else if (num_bits == 5) {
        merge_implicants_bits5(implicants, primes, input_index, output_index, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 64) { // implicants do not fit into one register, and we use the largest register size
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
        } else { // implicants that are compared fit into one 64-bit register
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
    }
}

static inline void merge_avx2_single_register(int bit_difference, __m256i impl1, __m256i primes1, __m128i *result, __m256i *primes_result) {
    assert(0 <= bit_difference && bit_difference <= 7);

    int block_len = 1 << bit_difference;

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
    // block_len <= 64: we can shift and compare without crossing 128-bit boundaries
    __m256i impl2;
    if (block_len == 64) {
        // we need to shift across 64-bit boundaries which needs an immediate value
        impl2 = _mm256_srli_si256(impl1, 8); // 8 bytes = 64 bits
    } else {
        impl2 = _mm256_srli_epi64(impl1, block_len);
    }

    __m256i aggregated = _mm256_and_si256(impl1, impl2);
    __m256i initial_result = _mm256_set1_epi64x(0); // prevent unitialized warnings
    __m256i shifted = _mm256_set1_epi64x(0);
    if (block_len == 1) {
        aggregated = _mm256_and_si256(aggregated, _mm256_set1_epi8(0b01010101));
        initial_result = aggregated;
        shifted = _mm256_srli_epi64(aggregated, 1);
    }
    if (block_len <= 2) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00110011));
        if (block_len == 2) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 2);
    }
    if (block_len <= 4) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi8(0b00001111));
        if (block_len == 4) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 4);
    }
    if (block_len <= 8) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi16(0x00FF));
        if (block_len == 8) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 8);
    }
    if (block_len <= 16) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi32(0x0000FFFF));
        if (block_len == 16) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_epi64(aggregated, 16);
    }
    if (block_len <= 32) {
        aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set1_epi64x(0x00000000FFFFFFFF));
        if (block_len == 32) {
            initial_result = aggregated;
        }
        shifted = _mm256_srli_si256(aggregated, 4); // 4 bytes = 32 bits
    }
    aggregated = _mm256_and_si256(_mm256_or_si256(aggregated, shifted), _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
    if (block_len == 64) {
        initial_result = aggregated;
    }
    // move 64-bit value aggregated[2] to result256[1] so result is in lower half
    __m256i result256 = _mm256_permute4x64_epi64(aggregated, 0b00001000);
    *result = _mm256_castsi256_si128(result256);

    // shift initial_result left by block_len
    __m256i shifted_initial_result;
    if (block_len == 64) {
        shifted_initial_result = _mm256_slli_si256(initial_result, 8); // 8 bytes = 64 bits
    } else {
        shifted_initial_result = _mm256_slli_epi64(initial_result, block_len);
    }
    __m256i merged = _mm256_or_si256(shifted_initial_result, initial_result);
    __m256i r_ = _mm256_andnot_si256(merged, primes1);
    *primes_result = r_;
}

void merge_implicants_avx2_single_pass(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits <= 7) {
        merge_implicants_avx2_single_pass_small_n(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 256) { // implicants do not fit into one register, and we use the largest register size
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
        } else { // implicants that are compared fit into one 256-bit register, i.e. block_len <= 128
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
                __m128i impl_result;
                __m256i primes_result;
                merge_avx2_single_register(i, impl1, primes1, &impl_result, &primes_result);
                _mm256_store_si256((__m256i*)(primes.bits + idx1 / 8), primes_result);
                if (i >= first_difference) {
                    _mm_store_si128((__m128i*)(implicants.bits + o_idx / 8), impl_result);
                    o_idx += 128;
                }
                idx1 += 256;
            }
        }
    }
}

prime_implicant_result prime_implicants_avx2_single_pass(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }

    uint64_t num_ops = 0;
    init_tsc();
    uint64_t counter_start = start_tsc();

    size_t input_index = 0;
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        ITT_START_TASK_NBITS(num_dashes);
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            merge_implicants_avx2_single_pass(implicants, primes, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }
        ITT_END_TASK();
#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    // mark last implicant prime if it is true
    BITMAP_SET(primes, num_implicants-1, BITMAP_CHECK(implicants, num_implicants-1));

    uint64_t cycles = stop_tsc(counter_start);
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
#ifdef COUNT_OPS
        .num_ops = num_ops,
#endif
    };
    return result;
}
#endif
