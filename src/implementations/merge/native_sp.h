#include "../../bitmap.h"

#if defined(__AVX2__)
#include "avx2_sp.h"
#elif defined(__aarch64__)
#include "neon_sp.h"
#else
#include "bits_sp.h"
#endif

static inline void merge_native_sp(bitmap implicants, bitmap primes, size_t input_index, size_t output_index, int num_bits, int first_difference) {
    #if defined(__AVX2__)
        merge_avx2_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
    #elif defined(__aarch64__)
        merge_neon_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
    #else
        merge_bits_sp(implicants, primes, input_index, output_index, num_bits, first_difference);
    #endif
}
