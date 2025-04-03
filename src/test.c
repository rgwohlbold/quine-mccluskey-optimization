#include "test.h"

#include <stdbool.h>
#include <stdlib.h>

#include "debug.h"
#include "dense.h"
#include "implicant.h"
#include "sparse.h"

typedef implicant (*implementation_function)(int num_bits, int num_trues, int *trues, int num_dont_cares,
                                             int *dont_cares, int *num_prime_implicants);

typedef struct {
    const char *name;
    implementation_function implementation;
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