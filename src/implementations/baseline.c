#include "baseline.h"

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

void merge_implicants_baseline(bool *implicants, bool *output, bool *merged, int num_bits, int first_difference) {
    // check all minterms that differ in the ith bit
    int o_idx = 0;
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);
        for (int block = 0; block < num_blocks; block++) {
            int idx1 = 2 * block * block_len;
            int idx2 = 2 * block * block_len + block_len;
            for (int k = 0; k < block_len; k++) {
                bool impl1 = implicants[idx1];
                bool impl2 = implicants[idx2];
                bool merged1 = merged[idx1];
                bool merged2 = merged[idx2];
                bool res = impl1 && impl2;
                bool merged1_ = merged1 || res;
                bool merged2_ = merged2 || res;

                merged[idx1] = merged1_;
                merged[idx2] = merged2_;

                // we don't want implicants in the output for which i < first_difference.
                // however, we still need to set the merged flag as we otherwise might
                // implicants prime that are implicitly considered in other calls.
                if (i >= first_difference) {
                    // o_idx = ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    output[o_idx++] = res;
                }
                idx1++;
                idx2++;
            }
        }
    }
}

prime_implicant_result prime_implicants_baseline(int num_bits, int num_trues, int *trues) {
    int num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bool *implicants = allocate_boolean_array(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        implicants[trues[i]] = true;
    }

    bool *merged_implicants = allocate_boolean_array(num_implicants);  // will initialize to false

    init_tsc();
    uint64_t counter_start = start_tsc();

    // Step 1: Merge all implicants iteratively, setting merged flags
    size_t input_index = 0;
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        // for each combination of num_dashes in num_bits, we need to run the algorithm once
        int iterations = binomial_coefficient(num_bits, num_dashes);
        // each input table has 2**remaining_bits elements
        int input_elements = 1 << remaining_bits;
        // each output table has 2**(remaining_bits-1) elements
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;

        // since we don't want any duplicates, make subsequent calls start at higher and higher bits
        // we need to adjust the number of output tables for this

        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            merge_implicants_baseline(&implicants[input_index], &implicants[output_index], &merged_implicants[input_index], remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }

    }
    // Step 2: Scan for unmerged implicants
    for (int i = 0; i < num_implicants; i++) {
        if (implicants[i] && !merged_implicants[i]) {
            BITMAP_SET_TRUE(primes, i);
        }
    }


    uint64_t cycles = stop_tsc(counter_start);

    free(implicants);
    free(merged_implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,

    };
    return result;
}
