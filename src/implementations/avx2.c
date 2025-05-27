#ifdef __AVX2__
#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../vtune.h"

#include "merge/avx2.h"

prime_implicant_result prime_implicants_avx2(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }
    bitmap merged = bitmap_allocate(num_implicants);

    uint64_t num_ops = 0;
    init_tsc();
    uint64_t counter_start = start_tsc();

    size_t input_index = 0;
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        ITT_START_TASK_SECTION(num_dashes);
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            merge_avx2(implicants, merged, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }
        ITT_END_TASK();
#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    ITT_START_GATHER_TASK();
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
    ITT_END_TASK();
#ifdef COUNT_OPS
        num_ops += 2 * num_implicants;
#endif

    uint64_t cycles = stop_tsc(counter_start);
    bitmap_free(implicants);
    bitmap_free(merged);

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
