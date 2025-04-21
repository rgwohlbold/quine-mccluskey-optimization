#include "sparse.h"

#include <stdbool.h>
#include <stdlib.h>

#include "util.h"

void minterm_number_to_implicant(int num_bits, int minterm, implicant dest) {
    for (int i = 0; i < num_bits; i++) {
        ternary_value val = TV_FALSE;
        if ((minterm & (1 << (num_bits - i - 1))) != 0) {
            val = TV_TRUE;
        }
        dest[i] = val;
    }
}

bool check_implicants_merge(int num_bits, implicant implicant1, implicant implicant2, int *difference_index) {
    // check that dashes align
    // TODO: can be optimized as implicants have the same number of dashes in our case
    for (int i = 0; i < num_bits; i++) {
        if ((implicant1[i] == TV_DASH && implicant2[i] != TV_DASH) ||
            (implicant1[i] != TV_DASH && implicant2[i] == TV_DASH)) {
            return false;
        }
    }
    // check minterm difference
    int difference = 0;
    for (int i = 0; i < num_bits; i++) {
        if ((implicant1[i] == TV_TRUE && implicant2[i] == TV_FALSE) ||
            (implicant1[i] == TV_FALSE && implicant2[i] == TV_TRUE)) {
            difference++;
            *difference_index = i;
        }
    }
    return difference == 1;
}

void merge_implicants(int num_bits, implicant implicant1, implicant dest, int difference_index) {
    for (int i = 0; i < num_bits; i++) {
        dest[i] = implicant1[i];
    }
    dest[difference_index] = TV_DASH;
}

void print_implicants(int num_bits, int num_implicants, implicant arr, char *msg) {
    printf("%s\n", msg);
    for (int i = 0; i < num_implicants; i++) {
        print_implicant(&arr[i * num_bits], num_bits);
    }
}

bool check_elt_in_implicant_list(int num_bits, implicant needle, ternary_value *haystack, int num_implicants) {
    for (int i = 0; i < num_implicants; i++) {
        bool all_match = true;
        for (int j = 0; j < num_bits; j++) {
            if (haystack[i * num_bits + j] != needle[j]) {
                all_match = false;
                break;
            }
        }
        if (all_match) return 1;
    }
    return 0;
}

/**
 * Compute prime implicants of the specified function
 *
 * values is a pointer to an array of 2**num_bits function values.
 * TODO: implement performance counters
 */
prime_implicant_result prime_implicants_sparse(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares) {
    int num_prime_implicants = 0;
    // a minterm is an array of num_bits ternary_value
    // to have space for all minterms, we allocate 3**num_bits * num_bits * sizeof(ternary_value)
    int num_implicants = calculate_num_implicants(num_bits);
    implicant uncombined = allocate_minterm_array(num_bits, num_implicants);
    implicant combined = allocate_minterm_array(num_bits, num_implicants);
    implicant primes = allocate_minterm_array(num_bits, num_trues+num_dont_cares);
    bool *merged = allocate_boolean_array(num_implicants);

    int num_uncombined_implicants = 0;
    for (int k = 0; k < num_trues; k++) {
        minterm_number_to_implicant(num_bits, trues[k], &uncombined[num_bits * num_uncombined_implicants]);
        num_uncombined_implicants++;
    }
    // TODO: check what exactly to do with don't cares here
    for (int k = 0; k < num_dont_cares; k++) {
        minterm_number_to_implicant(num_bits, dont_cares[k], &uncombined[num_bits * num_uncombined_implicants]);
        num_uncombined_implicants++;
    }

    while (true) {
        int num_combined_implicants = 0;
        for (int i = 0; i < num_uncombined_implicants; i++) {
            merged[i] = false;
        }

        for (int i = 0; i < num_uncombined_implicants; i++) {
            for (int k = i + 1; k < num_uncombined_implicants; k++) {
                implicant implicant1 = &uncombined[num_bits * i];
                implicant implicant2 = &uncombined[num_bits * k];
                int differerence_index = -1;
                if (check_implicants_merge(num_bits, implicant1, implicant2, &differerence_index)) {
                    merged[i] = true;
                    merged[k] = true;

                    ternary_value previous_value1 = implicant1[differerence_index];
                    implicant1[differerence_index] = TV_DASH;
                    bool in_list = check_elt_in_implicant_list(num_bits, implicant1, combined, num_combined_implicants);
                    implicant1[differerence_index] = previous_value1;
                    if (in_list) {
                        continue;
                    }
                    merge_implicants(num_bits, implicant1, &combined[num_bits * num_combined_implicants],
                                     differerence_index);
                    num_combined_implicants++;
                }
            }
        }

        for (int i = 0; i < num_uncombined_implicants; i++) {
            if (!merged[i]) {
                if (check_elt_in_implicant_list(num_bits, &uncombined[num_bits * i], primes, num_prime_implicants)) {
                    continue;
                }
                for (int k = 0; k < num_bits; k++) {
                    primes[num_bits * num_prime_implicants + k] = uncombined[num_bits * i + k];
                }
                num_prime_implicants++;
            }
        }

        if (num_combined_implicants == 0) {
            free(combined);
            free(uncombined);
            free(merged);
            prime_implicant_result result = {
                .primes = primes,
                .num_implicants = num_prime_implicants,
                .cycles = 0,
            };
            return result;
        }

        implicant tmp = uncombined;
        uncombined = combined;
        combined = tmp;
        num_uncombined_implicants = num_combined_implicants;
    }
}