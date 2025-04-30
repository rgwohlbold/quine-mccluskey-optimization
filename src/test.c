#include "test.h"

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#include "debug.h"
#include "dense.h"
#include "implicant.h"
#include "sparse.h"
#include "tsc_x86.h"
#include "util.h"

const prime_implicant_implementation implementations[] = {
    {"prime_implicants_dense", prime_implicants_dense},
    {"prime_implicants_sparse", prime_implicants_sparse},
};

typedef struct {
    const char *name;  // static storage
    int num_bits;
    int num_trues;
    int *trues;  // malloc'd
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

test_case make_test(const char *name, int num_bits, int num_trues, int *trues, int num_prime_implicants, char **prime_implicants) {
    int *new_trues = calloc(num_trues, sizeof(int));
    implicant new_prime_implicants = calloc(num_bits * num_prime_implicants, sizeof(ternary_value));
    if (new_trues == NULL || new_prime_implicants == NULL) {
        perror("could not allocate test case array");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_trues; i++) {
        new_trues[i] = trues[i];
    }
    for (int i = 0; i < num_prime_implicants; i++) {
        parse_implicant(prime_implicants[i], &new_prime_implicants[i * num_bits]);
    }
    test_case result = {name,
                        num_bits,
                        num_trues,
                        new_trues,
                        num_prime_implicants,
                        new_prime_implicants};
    return result;
}

void free_test(test_case test) {
    free(test.trues);
    free(test.prime_implicants);
}

#define MAKE_TEST(name, num_bits, trues_arr, prime_implicants_arr)         \
    make_test((name), (num_bits), sizeof(trues_arr) / sizeof((trues_arr)[0]), (trues_arr), \
              sizeof(prime_implicants_arr) / sizeof((prime_implicants_arr)[0]), (prime_implicants_arr));

test_case wikipedia_test() {
    int trues[] = {4, 8, 9, 10, 11, 12, 14, 15};
    char *prime_implicants[] = {"-100", "10--", "1--0", "1-1-"};
    return MAKE_TEST("wikipedia example", 4, trues, prime_implicants);
}

test_case wikipedia_test_2() {
    int trues[] = {4, 8, 10, 11, 12, 15};
    char *prime_implicants[] = {"10-0", "1-00", "101-", "1-11", "-100"};
    return MAKE_TEST("wikipedia example, no don't cares", 4, trues, prime_implicants);
}

test_case no_minterms() {
    int trues[] = {};
    char *prime_implicants[] = {};
    return MAKE_TEST("no minterms", 10, trues, prime_implicants);
}

test_case all_minterms() {
    int trues[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    char *prime_implicants[] = {"----"};
    return MAKE_TEST("all minterms", 4, trues, prime_implicants);
}

test_case half_minterms() {
    int trues[] = {0, 1, 2, 3, 4, 5, 6, 7};
    char *prime_implicants[] = {"0---"};
    return MAKE_TEST("half minterms", 4, trues, prime_implicants);
}

test_case other_half_minterms() {
    int trues[] = {
        992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
        1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023};
    char *prime_implicants[] = {"11111-----"};
    return MAKE_TEST("upper minterms", 10, trues, prime_implicants);
}

void test_implementations() {
    test_case test_cases[] = {
        wikipedia_test(), no_minterms(), half_minterms(), other_half_minterms(), all_minterms(),
    };
    for (unsigned long i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        test_case test = test_cases[i];
        for (unsigned long k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
            prime_implicant_implementation impl = implementations[k];

            LOG_INFO("checking '%s' -> '%s'", test.name, impl.name);

            prime_implicant_result result = impl.implementation(test.num_bits, test.num_trues, test.trues);
            int num_prime_implicants = result.num_implicants;
            implicant primes = result.primes;

            if (num_prime_implicants != test.num_prime_implicants) {
                LOG_INFO("wrong number of prime implicants: expected %d but got %d", test.num_prime_implicants,
                         num_prime_implicants);
            }
            for (int p = 0; p < num_prime_implicants; p++) {
                bool expected = check_elt_in_implicant_list(test.num_bits, &primes[p * test.num_bits],
                                                            test.prime_implicants, test.num_prime_implicants);
                if (!expected) {
                    LOG_INFO("implementation returned implicant that was not expected by test case:");
                    LOG_INFO_IMP(&primes[p * test.num_bits], test.num_bits);
                }
            }
            for (int p = 0; p < test.num_prime_implicants; p++) {
                bool expected = check_elt_in_implicant_list(test.num_bits, &test.prime_implicants[p * test.num_bits],
                                                            primes, num_prime_implicants);
                if (!expected) {
                    LOG_INFO("test case expected implicant that was not returned by implementation:");
                    LOG_INFO_IMP(&test.prime_implicants[p * test.num_bits], test.num_bits);
                }
            }
            free(primes);
        }
    }
    for (unsigned long i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        free_test(test_cases[i]);
    }
}

// for now, call all implementations on empty input and see performance
void measure_implementations(const char *implementation_name, int num_bits) {
    prime_implicant_implementation impl;
    bool implementation_found = false;
    for (unsigned long k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
        impl = implementations[k];
        if (strcmp(impl.name, implementation_name) == 0) {
            implementation_found = true;
            break;
        }
    }
    if (!implementation_found) {
        LOG_INFO("could not find implementation %s", implementation_name);
        return;
    }

    int trues[] = {};

    // warmup iteration
    impl.implementation(num_bits, 0, trues);

    LOG_INFO("measuring '%s' bits=%d", impl.name, num_bits);
    prime_implicant_result result = impl.implementation(num_bits, 0, trues);
    uint64_t cycles = result.cycles;
#ifdef COUNT_OPS
    uint64_t ops = result.num_ops;
#else
    uint64_t ops = 0;
#endif
    FILE *f = fopen("measurements.csv", "a");
    fprintf(f, "%s,%d,%lu,%lu\n", impl.name, num_bits, cycles, ops);
    fclose(f);

    free(result.primes);
}

void measure_merge(int num_bits) {
    LOG_INFO("measuring merge_implicants_dense bits=%d", num_bits);

    int input_elements = 1 << num_bits;
    int output_elements = num_bits << (num_bits-1);
    const int warmup_iterations = 10;
    bool *input_warmup = allocate_boolean_array(input_elements);
    bool *merged_warmup = allocate_boolean_array(input_elements);
    bool *output_warmup = allocate_boolean_array(output_elements);

    for (int i = 0; i < warmup_iterations; i++) {
        // use first difference 0 for now
      merge_implicants_dense(input_warmup, output_warmup, merged_warmup, num_bits, 0);
    }

    bool *input = allocate_boolean_array(input_elements);
    bool *merged = allocate_boolean_array(input_elements);
    bool *output = allocate_boolean_array(output_elements);
    // TODO: check if this makes a difference
    flush_cache(input, input_elements);
    flush_cache(merged, input_elements);
    flush_cache(output, output_elements);

    init_tsc();
    uint64_t counter = start_tsc();
    merge_implicants_dense(input, output, merged, num_bits, 0);
    uint64_t cycles = stop_tsc(counter);

    uint64_t num_ops = 3 * num_bits * (1 << (num_bits - 1));
    FILE *f = fopen("measurements_merge.csv", "a");
    fprintf(f, "%s,%d,%lu,%lu\n", "merge_implicants_dense", num_bits, cycles, num_ops);
    fclose(f);

}