#include "dense.h"

#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "tsc_x86.h"

void merge_implicants_dense(bool *implicants, bool *output, bool *merged, int num_bits, int first_difference) {
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

// calculate all binomials for k=0...n. binomials is an array of size (n+1)
void calculate_binomials(int n, int *binomials) {
    for (int k = 0; k <= n; k++) {
        if (k > n - k) {
            break;
        }
        unsigned long long res = 1;
        for (int i = 0; i < k; i++) {
            res = res * (n - i) / (i + 1);
        }
        binomials[k] = res;
        binomials[n - k] = res;
    }
}

// inefficient implementation to convert dash position number to dashes in res
void put_implicant_from_iteration(int num_bits, int num_dashes, int iteration, int value, implicant res) {
    // put all dashes to the back at first
    for (int k = 0; k < num_dashes; k++) {
        res[num_bits - k - 1] = TV_DASH;
    }
    for (int k = 0; k < num_bits - num_dashes; k++) {
        res[k] = TV_FALSE;
    }
    // move dashes around (iteration) times
    for (int k = 0; k < iteration; k++) {
        int leading_dashes = 0;
        while (res[leading_dashes] == TV_DASH) {
            res[leading_dashes] = TV_FALSE;
            leading_dashes++;
        }
        for (int j = 0; j < num_bits; j++) {
            if (res[j] != TV_DASH && res[j + 1] == TV_DASH) {
                res[j] = TV_DASH;
                res[j + 1] = TV_FALSE;
                while (leading_dashes > 0) {
                    j--;
                    res[j] = TV_DASH;
                    leading_dashes--;
                }
                break;
            }
        }
    }
    // put in value of implicant except for dashes
    int bit = 0;
    for (int k = 0; k < num_bits; k++) {
        int bitmask = 1 << bit;
        if (res[num_bits - k - 1] != TV_DASH) {
            if ((value & bitmask) != 0) {
                res[num_bits - k - 1] = TV_TRUE;
            } else {
                res[num_bits - k - 1] = TV_FALSE;
            }
            bit++;
        }
    }
}

prime_implicant_result prime_implicants_dense(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares) {
    int num_implicants = calculate_num_implicants(num_bits);
    implicant primes = allocate_minterm_array(num_bits, num_trues + num_dont_cares);

    int binomials[num_bits + 1];
    calculate_binomials(num_bits, binomials);

    bool *implicants = allocate_boolean_array(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        implicants[trues[i]] = true;
    }
    for (int i = 0; i < num_dont_cares; i++) {
        implicants[dont_cares[i]] = true;
    }

    bool *merged_implicants = allocate_boolean_array(num_implicants);  // will initialize to false

    uint64_t num_ops = 0;
    init_tsc();
    uint64_t counter_start = start_tsc();

    // Step 1: Merge all implicants iteratively, setting merged flags
    bool *input = &implicants[0];
    bool *merged = &merged_implicants[0];
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        // for each combination of num_dashes in num_bits, we need to run the algorithm once
        int iterations = binomials[num_dashes];
        // each input table has 2**remaining_bits elements
        int input_elements = 1 << remaining_bits;
        // each output table has 2**(remaining_bits-1) elements
        int output_elements = 1 << (remaining_bits - 1);

        bool *output = &input[iterations * input_elements];

        // since we don't want any duplicates, make subsequent calls start at higher and higher bits
        // we need to adjust the number of output tables for this

        int first_difference = 0;
        int min_first_difference = 0;
        for (int i = 0; i < iterations; i++) {
            merge_implicants_dense(input, output, merged, remaining_bits, first_difference);
            output = &output[(remaining_bits - first_difference) * output_elements];
            input = &input[input_elements];
            merged = &merged[input_elements];
            if (first_difference == remaining_bits) {
                min_first_difference++;
                first_difference = min_first_difference;
            } else {
                first_difference++;
            }
        }
#ifdef COUNT_OPS
        num_ops += 3 * iterations * remaining_bits * (1 << (remaining_bits - 1));
#endif
    }

    int num_prime_implicants = 0;
    uint64_t cycles = stop_tsc(counter_start);

    // Step 2: Scan for unmerged implicants
    input = &implicants[0];
    merged = &merged_implicants[0];
    for (int num_dashes = 0; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        // for each combination of num_dashes, we need to run the algorithm once
        int iterations = binomials[num_dashes];
        // each input table has 2**remaining_bits elements
        int input_elements = 1 << remaining_bits;

        for (int i = 0; i < iterations; i++) {
            for (int k = 0; k < input_elements; k++) {
                if (input[i * input_elements + k] && !merged[i * input_elements + k]) {
                    put_implicant_from_iteration(num_bits, num_dashes, i, k, &primes[num_prime_implicants * num_bits]);
                    num_prime_implicants++;
                }
            }
        }
        input = &input[iterations * input_elements];
        merged = &merged[iterations * input_elements];
    }
    free(implicants);
    free(merged_implicants);

    prime_implicant_result result = {
        .primes = primes,
        .num_implicants = num_prime_implicants,
        .cycles = cycles,
#ifdef COUNT_OPS
        .num_ops = num_ops,
#endif
    };
    return result;
}