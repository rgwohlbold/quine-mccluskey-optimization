#ifdef __AVX2__
#include "merge/avx2.h"

static inline void reduce_avx2(size_t num_implicants, bitmap implicants, bitmap merged, bitmap primes) {
    // Step 2: Scan for unmerged implicants
    for (size_t i = 0; i < num_implicants - (num_implicants % 256); i += 256) {
        __m256i implicant_true = _mm256_load_si256((__m256i*)(implicants.bits + i / 8));
        __m256i merged_true = _mm256_load_si256((__m256i*)(merged.bits + i / 8));
        __m256i prime_true = _mm256_andnot_si256(merged_true, implicant_true);
        _mm256_store_si256((__m256i*)(primes.bits + i / 8), prime_true);
    }
    for (size_t i = num_implicants - (num_implicants % 256); i < num_implicants - num_implicants % 64; i += 64) {
        uint64_t implicant_true = ((uint64_t*)implicants.bits)[i / 64];
        uint64_t merged_true = ((uint64_t*)merged.bits)[i / 64];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t*)primes.bits)[i / 64] = prime_true;
    }
    for (size_t i = num_implicants - (num_implicants % 64); i < num_implicants; i++) {
        if (BITMAP_CHECK(implicants, i) && !BITMAP_CHECK(merged, i)) {
            BITMAP_SET_TRUE(primes, i);
        }
    }
}

#define IMPLEMENTATION_FUNCTION prime_implicants_avx2
#define MERGE_FUNCTION merge_avx2
#define REDUCE_FUNCTION reduce_avx2

#include "algorithms/base.h"

#endif
