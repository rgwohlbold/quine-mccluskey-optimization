#include "merge/bits.h"

static inline void reduce_bits(size_t num_implicants, bitmap implicants, bitmap merged, bitmap primes) {
    for (size_t i = 0; i < num_implicants / 64; i++) {
        uint64_t implicant_true = ((uint64_t*)implicants.bits)[i];
        uint64_t merged_true = ((uint64_t*)merged.bits)[i];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t*)primes.bits)[i] = prime_true;
    }
    for (size_t i = num_implicants - (num_implicants % 64); i < num_implicants; i++) {
        if (BITMAP_CHECK(implicants, i) && !BITMAP_CHECK(merged, i)) {
            BITMAP_SET_TRUE(primes, i);
        }
    }
}

#define IMPLEMENTATION_FUNCTION prime_implicants_bits
#define MERGE_FUNCTION merge_bits
#define REDUCE_FUNCTION reduce_bits

#include "algorithms/base.h"
