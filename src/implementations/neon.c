#ifdef __aarch64__
#include "merge/neon.h"

static inline void reduce_neon(size_t num_implicants, bitmap implicants, bitmap merged, bitmap primes) {
    for (size_t i = 0; i < num_implicants - (num_implicants % 128); i += 128) {
        uint64x2_t impl = vld1q_u64((uint64_t*)(implicants.bits + i/8));
        uint64x2_t mer = vld1q_u64((uint64_t*)(merged.bits + i/8));
        uint64x2_t prime = vbicq_u64(impl, mer);
        vst1q_u64((uint64_t*)(primes.bits + i/8), prime);

    }
    for (size_t i = num_implicants - (num_implicants % 128); i < num_implicants - num_implicants % 64; i += 64) {
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

#define IMPLEMENTATION_FUNCTION prime_implicants_neon
#define MERGE_FUNCTION merge_neon
#define REDUCE_FUNCTION reduce_neon

#include "algorithms/base.h"

#endif
