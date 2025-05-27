#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "../util.h"
#include "../my_signpost.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif

#include "merge/bits_sp.h"

prime_implicant_result prime_implicants_bits_single_pass(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);
    bitmap implicants = bitmap_allocate(num_implicants);
    SIGNPOST_INIT();
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }

    uint64_t num_ops = 0;
    init_tsc();
    uint64_t counter_start = start_tsc();

    size_t input_index = 0;
    // LOG_DEBUG("Hi");
    SIGNPOST_INTERVAL_BEGIN(gLog, gSpid, "all_dashes", "Metadata: %s", "Foo");
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        // LOG_DEBUG("num_dashes: %d", num_dashes);
        // print_bitmap_sparse("Implicants", &implicants);
        // print_bitmap_sparse("primes", &primes);
        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            merge_bits_sp(implicants, primes, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }


#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }
    SIGNPOST_INTERVAL_END(gLog, gSpid, "all_dashes", "");
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
