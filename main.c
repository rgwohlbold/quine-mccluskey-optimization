#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 1

typedef enum {
    FV_FALSE,
    FV_TRUE,
    FV_DONT_CARE,
} function_value;

typedef enum {
    TV_FALSE,
    TV_TRUE,
    TV_DASH,
} ternary_value;

typedef ternary_value *implicant;

implicant allocate_minterm_array(int num_bits) {
    implicant minterms = (ternary_value *)calloc((1 << num_bits) * num_bits, sizeof(ternary_value));
    if (minterms == NULL) {
        perror("could not allocate minterms array");
        exit(EXIT_FAILURE);
    }
    return minterms;
}

bool *allocate_boolean_array(int num_bits) {
    bool *arr = (bool *)calloc(1 << num_bits, sizeof(bool));
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
        for (int k = 0; k < num_bits; k++) {
            ternary_value val = arr[i * num_bits + k];
            if (val == TV_TRUE) {
                printf("1");
            } else if (val == TV_FALSE) {
                printf("0");
            } else if (val == TV_DASH) {
                printf("-");
            }
        }
        printf("\n");
    }
}

/**
 * Compute prime implicants of the specified function
 *
 * values is a pointer to an array of 2**num_bits function values.
 * TODO: specify return value
 */
implicant prime_implicants(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares,
                           int *num_prime_implicants) {
    // a minterm is an array of num_bits ternary_value
    // to have space for all minterms, we allocate 2**num_bits * num_bits * sizeof(ternary_value)
    implicant uncombined = allocate_minterm_array(num_bits);
    implicant combined = allocate_minterm_array(num_bits);
    implicant primes = allocate_minterm_array(num_bits);
    bool *merged = allocate_boolean_array(num_bits);
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
#if DEBUG
    print_implicants(num_bits, num_uncombined_implicants, uncombined, "initial uncombined implicants");
#endif

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

int main(int argc, char **argv) {
    int num_bits = 4;
    int trues[] = {4, 8, 10, 11, 12, 15};
    int dont_cares[] = {9, 14};
    // function_value *example_function = build_table(4, sizeof(trues) / sizeof(trues[0]), trues, sizeof(dont_cares) /
    // sizeof(dont_cares[0]), dont_cares); print_table(num_bits, example_function); free(example_function);
    int num_prime_implicants = 0;
    implicant primes = prime_implicants(num_bits, sizeof(trues) / sizeof(trues[0]), trues,
                                        sizeof(dont_cares) / sizeof(dont_cares[0]), dont_cares, &num_prime_implicants);
    print_implicants(num_bits, num_prime_implicants, primes, "prime implicants");
    free(primes);
}