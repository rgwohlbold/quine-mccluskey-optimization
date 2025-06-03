#pragma once

#ifndef __BMI2__
#error "need BMI2 as base case to avx512_single_pass"
#endif

#include <assert.h>
#include <immintrin.h>
#include "../../bitmap.h"
#include "bits_sp.h"
#include "../../debug.h"
#include <stdio.h>
#include <stdint.h>

static void merge_avx512_sp_small_n_old_loop(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
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

// static void print_m512i_epi64(__m512i v, char* msg) {
//     __attribute__((aligned(64))) uint64_t val[8];
//     _mm512_store_si512(val, v);  // or _mm512_storeu_si512 for unaligned

//     LOG_DEBUG("%s", msg);
//     for (int i = 0; i < 8; ++i) {
//         LOG_DEBUG("val[%d] = 0x%016llx", i, (unsigned long long)val[i]);
//     }
//     LOG_DEBUG("");
// }
// static void print_m256i_epi64_(__m256i v, char* msg) {
//     __attribute__((aligned(32))) uint64_t val[4];
//     _mm256_store_epi64(val, v);

//     LOG_DEBUG("%s", msg);
//     for (int i = 0; i < 4; ++i) {
//         LOG_DEBUG("val[%d] = 0x%016llx", i, (unsigned long long)val[i]);
//     }
//     LOG_DEBUG("");
// }

static inline void merge_avx512_sp_single_register_old_loop(int bit_difference, __m512i impl1, __m512i primes1, __m256i *result, __m512i *primes_result) {
    assert(0 <= bit_difference && bit_difference <= 8);

    int block_len = 1 << bit_difference;


    if (block_len == 256) {
        __m512i indices = _mm512_set_epi64(7, 6, 5, 4, 7, 6, 5, 4);
        __m512i indices2 = _mm512_set_epi64(3, 2, 1, 0, 3, 2, 1, 0);
        
        // upper half -> lower half of impl1 (NOT set upper half to zero)
        __m512i impl1_shuffled = _mm512_permutexvar_epi64(indices, impl1);
        __m512i aggregated = _mm512_and_si512(impl1, impl1_shuffled);
        // set upper and lower halves to lower half of aggregated
        __m512i merged = _mm512_permutexvar_epi64(indices2, aggregated); //_mm256_permute2x128_si256(aggregated, aggregated, 0x00);
        
        // use lower half of aggregated
        *result = _mm512_castsi512_si256(aggregated);
        *primes_result = _mm512_andnot_si512(merged, primes1);
        return;
    }

    // block_len < 64: we can shift and compare using _mm512_set_epi64 - otherwise we need to permute
    __m512i impl2;
    if (block_len < 64) {
        impl2 = _mm512_srli_epi64(impl1, block_len);
    } else if (block_len == 64) {
        // "shift" across 64-bit boundaries
        __m512i indices = _mm512_set_epi64(7, 7, 5, 5, 3, 3, 1, 1); 
        impl2 = _mm512_permutexvar_epi64(indices, impl1);

    } else { //if (block_len == 128) 
        __m512i indices = _mm512_set_epi64(7, 6, 7, 6, 3, 2, 3, 2); 
        impl2 = _mm512_permutexvar_epi64(indices, impl1);
    }


    __m512i aggregated = _mm512_and_si512(impl1, impl2);
    __m512i initial_result = _mm512_setzero_si512(); // prevent unitialized warnings
    __m512i shifted = _mm512_setzero_si512();
    if (block_len == 1) {
        aggregated = _mm512_and_si512(aggregated, _mm512_set1_epi8(0b01010101));
        initial_result = aggregated;
        shifted = _mm512_srli_epi64(aggregated, 1);
    } 
    if (block_len <= 2) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set1_epi8(0b00110011));
        if (block_len == 2) {
            initial_result = aggregated;
        }
        shifted = _mm512_srli_epi64(aggregated, 2);
    }
    if (block_len <= 4) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set1_epi8(0b00001111));
        if (block_len == 4) {
            initial_result = aggregated;
        }
        shifted = _mm512_srli_epi64(aggregated, 4);
    }
    if (block_len <= 8) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set1_epi16(0x00FF));
        if (block_len == 8) {
            initial_result = aggregated;
        }
        shifted = _mm512_srli_epi64(aggregated, 8);
    }
    if (block_len <= 16) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set1_epi32(0x0000FFFF));
        if (block_len == 16) {
            initial_result = aggregated;
        }
        shifted = _mm512_srli_epi64(aggregated, 16);
    }
    if (block_len <= 32) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set1_epi64(0x00000000FFFFFFFF));
        if (block_len == 32) {
            initial_result = aggregated;
        }
        shifted = _mm512_alignr_epi8(_mm512_setzero_si512(), aggregated, 4);
    }
    
    if (block_len <= 64) {
        aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set_epi64(0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF));
        if (block_len == 64) {
            initial_result = aggregated;
        }
        __m512i indices = _mm512_set_epi64(7, 7, 6, 5, 4, 3, 2, 1);
        shifted = _mm512_permutexvar_epi64(indices, aggregated);
    }

    aggregated = _mm512_and_si512(_mm512_or_si512(aggregated, shifted), _mm512_set_epi64(0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF));
    if (block_len == 128) {
        initial_result = aggregated;
    }

    
    // move aggregated[5, 4] to result[1, 0] so the whole result is in lower half
    __m512i indices2 = _mm512_set_epi64(7, 6, 5, 4, 5, 4, 1, 0); 
    __m512i result512 = _mm512_permutexvar_epi64(indices2, aggregated);
    *result = _mm512_castsi512_si256(result512);

    // shift initial_result left by block_len
    __m512i shifted_initial_result;
    __m512i merged;
    if (block_len < 64) {
        shifted_initial_result = _mm512_slli_epi64(initial_result, block_len);
        merged = _mm512_or_si512(shifted_initial_result, initial_result);
    } else if (block_len == 64) {
        __m512i indices = _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0);
        merged = _mm512_permutexvar_epi64(indices, initial_result); 

    } else {
        // shift across 128-bit boundaries
        __m512i indices = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0); 
        merged = _mm512_permutexvar_epi64(indices, initial_result);
    }

    __m512i r_ = _mm512_andnot_si512(merged, primes1);
    *primes_result = r_;
}

static void merge_avx512_sp_old_loop(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    if (num_bits <= 8) {
        merge_avx512_sp_small_n_old_loop(implicants, primes, input_index, output_index, num_bits, first_difference);
        return;
    }
    size_t o_idx = output_index;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);

        if (block_len >= 512) { // implicants do not fit into one register, and we use the largest register size
            for (int block = 0; block < num_blocks; block++) {
                size_t idx1 = input_index + 2 * block * block_len;
                size_t idx2 = input_index + 2 * block * block_len + block_len;

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
        } else { // implicants that are compared fit into one 512-bit register, i.e. block_len <= 256
            for (int block = 0; block < num_blocks; block += 256 / block_len) {

                // LOG_DEBUG("block_len=%d, block=%d, num_blocks=%d, i=%d", block_len, block, num_blocks, i);

                size_t idx1 = input_index + 2 * block * block_len;

                __m512i impl1 = _mm512_load_si512((__m512i*)(implicants.bits + idx1 / 8));
                __m512i primes1;
                // assume we always run block_len=1 first and primes[idx1/8] is uninitialized at this point.
                // in that case, assume all implicants are prime
                if (block_len == 1) {
                    primes1 = impl1;
                } else {
                    primes1 = _mm512_load_si512((__m512i*)(primes.bits + idx1 / 8));
                }
                __m256i impl_result;
                __m512i primes_result;
                merge_avx512_sp_single_register_old_loop(i, impl1, primes1, &impl_result, &primes_result);
                _mm512_store_si512((__m512i*)(primes.bits + idx1 / 8), primes_result);
                if (i >= first_difference) {
                    _mm256_store_si256((__m256i*)(implicants.bits + o_idx / 8), impl_result);
                    o_idx += 256;
                }
                idx1 += 512;
            }
        }
    }
}
