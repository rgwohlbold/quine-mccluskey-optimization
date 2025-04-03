#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "implicant.h"

implicant allocate_minterm_array(int num_bits) {
    implicant minterms = (ternary_value *)calloc((1 << num_bits) * num_bits, sizeof(ternary_value));
    if (minterms == NULL) {
        perror("could not allocate minterms array");
        exit(EXIT_FAILURE);
    }
    return minterms;
}

bool *allocate_boolean_array(int num_elements) {
    bool *arr = (bool *)calloc(num_elements, sizeof(bool));
    if (arr == NULL) {
        perror("could not allocate boolean array");
        exit(EXIT_FAILURE);
    }
    return arr;
}

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
        if (implicant1[i] == TV_DASH && implicant2[i] != TV_DASH ||
            implicant1[i] != TV_DASH && implicant2[i] == TV_DASH) {
            return false;
        }
    }
    // check minterm difference
    int difference = 0;
    for (int i = 0; i < num_bits; i++) {
        if (implicant1[i] == TV_TRUE && implicant2[i] == TV_FALSE ||
            implicant1[i] == TV_FALSE && implicant2[i] == TV_TRUE) {
            difference++;
            *difference_index = i;
        }
    }
    return difference == 1;
}

void merge_implicants(int num_bits, implicant implicant1, implicant implicant2, implicant dest, int difference_index) {
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
 * TODO: specify return value
 */
implicant prime_implicants_sparse(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares,
                           int *num_prime_implicants) {
    // a minterm is an array of num_bits ternary_value
    // to have space for all minterms, we allocate 2**num_bits * num_bits * sizeof(ternary_value)
    implicant uncombined = allocate_minterm_array(num_bits);
    implicant combined = allocate_minterm_array(num_bits);
    implicant primes = allocate_minterm_array(num_bits);
    bool *merged = allocate_boolean_array(1 << num_bits);
    *num_prime_implicants = 0;

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
                    // TODO: duplicate detection
                    merge_implicants(num_bits, implicant1, implicant2, &combined[num_bits * num_combined_implicants],
                                     differerence_index);
                    num_combined_implicants++;
                    merged[i] = true;
                    merged[k] = true;
                }
            }
        }

        for (int i = 0; i < num_uncombined_implicants; i++) {
            if (!merged[i]) {
                // TODO: check that the implicant is not yet contained in primes
                if (check_elt_in_implicant_list(num_bits, &uncombined[num_bits * i], primes, *num_prime_implicants)) {
                    continue;
                }
                for (int k = 0; k < num_bits; k++) {
                    primes[num_bits * (*num_prime_implicants) + k] = uncombined[num_bits * i + k];
                }
                (*num_prime_implicants)++;
            }
        }

        if (num_combined_implicants == 0) {
            free(combined);
            free(uncombined);
            return primes;
        }

        implicant tmp = uncombined;
        uncombined = combined;
        combined = tmp;
        num_uncombined_implicants = num_combined_implicants;
    }
}

function_value *build_table(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares) {
    int num_values = 1 << num_bits;
    function_value *table = calloc(num_values, sizeof(function_value));
    if (table == NULL) {
        perror("could not allocate function value table");
        exit(EXIT_FAILURE);
    }
    /* this could be removed due to calloc() zero-initializing memory,
    but leaving it in for completeness' sake */
    for (int i = 0; i < num_values; i++) {
        table[i] = FV_FALSE;
    }
    for (int i = 0; i < num_trues; i++) {
        table[trues[i]] = FV_TRUE;
    }
    for (int i = 0; i < num_dont_cares; i++) {
        table[dont_cares[i]] = FV_DONT_CARE;
    }
    return table;
}

void print_table(int num_bits, function_value *table) {
    int num_values = 1 << num_bits;
    for (int i = 0; i < num_values; i++) {
        switch (table[i]) {
            case FV_TRUE:
                printf("%d\t1\n", i);
                break;
            case FV_FALSE:
                printf("%d\t0\n", i);
                break;
            case FV_DONT_CARE:
                printf("%d\tX\n", i);
                break;
        }
    }
}

void merge_implicants_dense(bool *implicants, bool *output, bool *merged, int num_bits, int first_difference) {
    // check all minterms that differ in the ith bit
    for (int i = 0; i < num_bits; i++) {
        int block_len = 1 << i;
        int num_blocks = 1 << (num_bits - i - 1);
        for (int block = 0; block < num_blocks; block++) {
            for (int k = 0; k < block_len; k++) {
                int idx1 = 2 * block * block_len + k;
                int idx2 = 2 * block * block_len + block_len + k;

                merged[idx1] = merged[idx1] || (implicants[idx1] && implicants[idx2]);
                merged[idx2] = merged[idx2] || (implicants[idx1] && implicants[idx2]);

                // we don't want implicants in the output for which i < first_difference.
                // however, we still need to set the merged flag as we otherwise might
                // implicants prime that are implicitly considered in other calls.
                if (i >= first_difference) {
                    int o_idx = ((i - first_difference) << (num_bits - 1)) + block * block_len + k;
                    output[o_idx] = implicants[idx1] && implicants[idx2];
                }
            }
        }
    }
}

// calculate 3**num_bits
int calculate_num_implicants(int num_bits) {
    int num_implicants = 1;
    for (int i = 0; i < num_bits; i++) {
        num_implicants *= 3;
    }
    return num_implicants;
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

implicant prime_implicants_dense(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares,
                                 int *num_prime_implicants) {
    implicant primes = allocate_minterm_array(num_bits);
    *num_prime_implicants = 0;

    int binomials[num_bits + 1];
    calculate_binomials(num_bits, binomials);

    int num_implicants = calculate_num_implicants(num_bits);
    bool *implicants = allocate_boolean_array(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        implicants[trues[i]] = true;
    }
    for (int i = 0; i < num_dont_cares; i++) {
        implicants[dont_cares[i]] = true;
    }

    bool *merged_implicants = allocate_boolean_array(num_implicants);  // will initialize to false

    // Step 1: Merge all implicants iteratively, setting merged flags
    bool *input = &implicants[0];
    bool *merged = &merged_implicants[0];
    for (int num_dashes = 0; num_dashes < num_bits; num_dashes++) {
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
        int k = 0;
        for (int i = 0; i < iterations; i++) {
            merge_implicants_dense(&input[i * input_elements], &output[k * output_elements],
                                   &merged[i * input_elements], remaining_bits, i);
            k += iterations - i - 1;
        }
        input = output;
        merged = &merged[iterations * input_elements];
    }

    // Step 2: Scan for unmerged implicants
    input = &implicants[0];
    merged = &merged_implicants[0];
    for (int num_dashes = 0; num_dashes < num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        // for each combination of num_dashes, we need to run the algorithm once
        int iterations = binomials[num_dashes];
        // each input table has 2**remaining_bits elements
        int input_elements = 1 << remaining_bits;

        for (int i = 0; i < iterations; i++) {
            for (int k = 0; k < input_elements; k++) {
                if (input[i * input_elements + k] && !merged[i * input_elements + k]) {
                    put_implicant_from_iteration(num_bits, num_dashes, i, k, &primes[*num_prime_implicants * num_bits]);
                    (*num_prime_implicants)++;
                }
            }
        }
        input = &input[iterations * input_elements];
        merged = &merged[iterations * input_elements];
    }
    free(implicants);
    free(merged_implicants);
    return primes;
}

typedef struct {
    const char *name;
    implicant (*implementation)(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares,
                                int *num_prime_implicants);
} prime_implicant_implementation;

const prime_implicant_implementation implementations[] = {
    {"prime_implicants_dense", prime_implicants_dense},
    {"prime_implicants_sparse", prime_implicants_sparse},
};

typedef struct {
    const char *name;  // static storage
    int num_bits;
    int num_trues;
    int *trues;  // malloc'd
    int num_dont_cares;
    int *dont_cares;  // malloc'd
    int num_prime_implicants;
    implicant prime_implicants;  // malloc'd
} test_case;

void parse_implicant(const char *s, implicant impl) {
    while (*s) {
        if (*s == '0') {
            *impl = TV_FALSE;
        } else if (*s == '1') {
            *impl = TV_TRUE;
        } else if (*s == '-') {
            *impl = TV_DASH;
        } else {
            fprintf(stderr, "could not parse implicant %s\n", s);
            exit(EXIT_FAILURE);
        }
        s++;
        impl++;
    }
}

test_case make_test(const char *name, int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares,
                    int num_prime_implicants, char **prime_implicants) {
    int *new_trues = calloc(num_trues, sizeof(int));
    int *new_dont_cares = calloc(num_dont_cares, sizeof(int));
    implicant new_prime_implicants = calloc(num_bits * num_prime_implicants, sizeof(ternary_value));
    if (new_trues == NULL || new_dont_cares == NULL || new_prime_implicants == NULL) {
        perror("could not allocate test case array");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_trues; i++) {
        new_trues[i] = trues[i];
    }
    for (int i = 0; i < num_dont_cares; i++) {
        new_dont_cares[i] = dont_cares[i];
    }
    for (int i = 0; i < num_prime_implicants; i++) {
        parse_implicant(prime_implicants[i], &new_prime_implicants[i * num_bits]);
    }
    test_case result = {name,
                        num_bits,
                        num_trues,
                        new_trues,
                        num_dont_cares,
                        new_dont_cares,
                        num_prime_implicants,
                        new_prime_implicants};
    return result;
}

void free_test(test_case test) {
    free(test.trues);
    free(test.dont_cares);
    free(test.prime_implicants);
}

#define MAKE_TEST(name, num_bits, trues_arr, dont_cares_arr, prime_implicants_arr)         \
    make_test((name), (num_bits), sizeof(trues_arr) / sizeof((trues_arr)[0]), (trues_arr), \
              sizeof(dont_cares_arr) / sizeof((dont_cares_arr)[0]), (dont_cares_arr),      \
              sizeof(prime_implicants_arr) / sizeof((prime_implicants_arr)[0]), (prime_implicants_arr));

test_case wikipedia_test() {
    int trues[] = {4, 8, 10, 11, 12, 15};
    int dont_cares[] = {9, 14};
    char *prime_implicants[] = {"-100", "10--", "1--0", "1-1-"};
    return MAKE_TEST("wikipedia example", 4, trues, dont_cares, prime_implicants);
}

test_case wikipedia_test_2() {
    int trues[] = {4, 8, 10, 11, 12, 15};
    int dont_cares[] = {};
    char *prime_implicants[] = {"10-0", "1-00", "101-", "1-11", "-100"};
    return MAKE_TEST("wikipedia example, no don't cares", 4, trues, dont_cares, prime_implicants);
}

test_case no_minterms() {
    int trues[] = {};
    int dont_cares[] = {};
    char *prime_implicants[] = {};
    return MAKE_TEST("no minterms", 10, trues, dont_cares, prime_implicants);
}

test_case all_minterms() {
    int trues[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int dont_cares[] = {};
    char *prime_implicants[] = {"----"};
    return MAKE_TEST("all minterms", 4, trues, dont_cares, prime_implicants);
}

test_case half_minterms() {
    int trues[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int dont_cares[] = {};
    char *prime_implicants[] = {"0---"};
    return MAKE_TEST("half minterms", 4, trues, dont_cares, prime_implicants);
}

void test_implementations() {
    test_case test_cases[] = {
        wikipedia_test(), wikipedia_test_2(), no_minterms(), half_minterms(), all_minterms(),
    };
    for (int i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        test_case test = test_cases[i];
        for (int k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
            prime_implicant_implementation impl = implementations[k];

            LOG_INFO("checking '%s' -> '%s'", test.name, impl.name);
    int num_prime_implicants = 0;
            implicant result = impl.implementation(test.num_bits, test.num_trues, test.trues, test.num_dont_cares,
                                                   test.dont_cares, &num_prime_implicants);
            if (num_prime_implicants != test.num_prime_implicants) {
                LOG_INFO("wrong number of prime implicants: expected %d but got %d", test.num_prime_implicants,
                         num_prime_implicants);
            }
            for (int p = 0; p < num_prime_implicants; p++) {
                bool expected = check_elt_in_implicant_list(test.num_bits, &result[p * test.num_bits],
                                                            test.prime_implicants, test.num_prime_implicants);
                if (!expected) {
                    LOG_INFO("implementation returned implicant that was not expected by test case:");
                    LOG_INFO_IMP(&result[p * test.num_bits], test.num_bits);
                }
            }
            for (int p = 0; p < test.num_prime_implicants; p++) {
                bool expected = check_elt_in_implicant_list(test.num_bits, &test.prime_implicants[p * test.num_bits],
                                                            result, num_prime_implicants);
                if (!expected) {
                    LOG_INFO("test case expected implicant that was not returned by implementation:");
                    LOG_INFO_IMP(&test.prime_implicants[p * test.num_bits], test.num_bits);
                }
            }
        }
    }
    for (int i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        free_test(test_cases[i]);
    }
}

int main(int argc, char **argv) {
    test_implementations();
}