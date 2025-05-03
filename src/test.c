#include "test.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "dense.h"
#include "implicant.h"
#ifdef __x86_64__
#include "tsc_x86.h"
#endif
#ifdef __aarch64__
#include "vct_arm.h"
#endif
#include "util.h"

const prime_implicant_implementation implementations[] = {
    {"prime_implicants_dense", prime_implicants_dense},
};

typedef struct {
    const char *name;  // static storage
    int num_bits;
    int num_trues;
    int *trues;  // malloc'd
    int num_prime_implicants;
    bitmap prime_implicants;
} test_case;

test_case make_test(const char *name, int num_bits, int num_trues, int *trues, int num_prime_implicants,
                    char **prime_implicants) {
    char *name_copy = malloc(strlen(name) + 1);
    if (name_copy == NULL) {
        perror("could not allocate name");
        exit(EXIT_FAILURE);
    }
    strncpy(name_copy, name, strlen(name) + 1);
    name_copy[strlen(name)] = '\0';  // null terminate
    int num_implicants = calculate_num_implicants(num_bits);
    int *new_trues = calloc(num_trues, sizeof(int));
    if (new_trues == NULL) {
        perror("could not allocate test case array");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_trues; i++) {
        new_trues[i] = trues[i];
    }
    bitmap new_prime_implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_prime_implicants; i++) {
        int bitset_index = bitmap_implicant_to_index(num_bits, prime_implicants[i]);
        BITMAP_SET_TRUE(new_prime_implicants, bitset_index);
    }
    test_case result = {name_copy, num_bits, num_trues, new_trues, num_prime_implicants, new_prime_implicants};
    return result;
}

void free_test(test_case test) {
    free(test.trues);
    bitmap_free(test.prime_implicants);
}

#define MAKE_TEST(name, num_bits, trues_arr, prime_implicants_arr)                         \
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

char *fgets_comments(char *str, int num, FILE *stream) {
    // Aux function same as fgets, but ignores the line if it starts with '#'
    char *result = fgets(str, num, stream);
    while (result != NULL && str[0] == '#') {
        result = fgets(str, num, stream);
    }
    return result;
}

void from_testfile(const char *filename, test_case *dest) {
    /**
     * Format:
     *  <name>
     *   <num_bits> <num_trues> <num_prime_implicants>
     *   <trues>... times num_trues
     *   <prime_implicants>... times num_prime_implicants
     *
     * Comments:
     *   <prime_implicants> is a string of '-' and '0'/'1' of length num_bits
     *   <trues> is a number between 0 and 2^num_bits-1
     */
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        perror("could not open test file");
        exit(EXIT_FAILURE);
    }
    char line[1024];
    int num_bits = 0;
    int num_trues = 0;
    int num_prime_implicants = 0;
    int *trues = NULL;
    char **prime_implicants = NULL;
    char *name = NULL;
    if (fgets_comments(line, sizeof(line), f) == NULL) {
        perror("could not read test file");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    line[strcspn(line, "\n")] = 0;  // remove newline
    name = malloc(strlen(line) + 1);
    if (name == NULL) {
        perror("could not allocate name");
        exit(EXIT_FAILURE);
    }
    strncpy(name, line, strlen(line) + 1);
    LOG_DEBUG("name=%s", name);
    if (fgets_comments(line, sizeof(line), f) == NULL) {
        perror("could not read test file");
        fclose(f);
        free(name);
        exit(EXIT_FAILURE);
    }
    sscanf(line, "%d %d %d", &num_bits, &num_trues, &num_prime_implicants);
    LOG_DEBUG("num_bits=%d num_trues=%d num_prime_implicants=%d", num_bits, num_trues, num_prime_implicants);
    trues = malloc(num_trues * sizeof(int));
    prime_implicants = malloc(num_prime_implicants * sizeof(char *));
    for (int i = 0; i < num_trues; i++) {
        fgets_comments(line, sizeof(line), f);
        sscanf(line, "%d", &trues[i]);
    }
    for (int i = 0; i < num_prime_implicants; i++) {
        fgets_comments(line, sizeof(line), f);
        line[strcspn(line, "\n")] = 0;  // remove newline
        prime_implicants[i] = malloc(num_bits + 1);
        if (prime_implicants[i] == NULL) {
            perror("could not allocate prime implicant");
            exit(EXIT_FAILURE);
        }
        strncpy(prime_implicants[i], line, num_bits);
        prime_implicants[i][num_bits] = '\0';  // null terminate
    }

    *dest = make_test(name, num_bits, num_trues, trues, num_prime_implicants, prime_implicants);

    free(trues);
    for (int i = 0; i < num_prime_implicants; i++) {
        free(prime_implicants[i]);
    }
    free(prime_implicants);
    free(name);
    fclose(f);
}

void test_implementations(char **testfiles, int num_testfiles) {
    test_case *test_cases = malloc(num_testfiles * sizeof(test_case));
    for (int i = 0; i < num_testfiles; i++) {
        from_testfile(testfiles[i], &test_cases[i]);
    }
    for (unsigned long i = 0; i < (unsigned long)num_testfiles; i++) {
        test_case test = test_cases[i];
        for (unsigned long k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
            prime_implicant_implementation impl = implementations[k];
            LOG_INFO("checking '%s' -> '%s'", test.name, impl.name);

            prime_implicant_result result = impl.implementation(test.num_bits, test.num_trues, test.trues);
            if (!bitmap_cmp(result.primes, test.prime_implicants)) {
                for (int i = 0; i < result.primes.num_bits; i++) {
                    if (BITMAP_CHECK(result.primes, i) && !BITMAP_CHECK(test.prime_implicants, i)) {
                        char s[test.num_bits + 1];
                        s[test.num_bits] = '\0';
                        bitmap_index_to_implicant(test.num_bits, i, s);
                        LOG_WARN("returned implicant %s (bitmap index %d) which was not expected by test case", s, i);
                    }
                    if (!BITMAP_CHECK(result.primes, i) && BITMAP_CHECK(test.prime_implicants, i)) {
                        char s[test.num_bits + 1];
                        s[test.num_bits] = '\0';
                        bitmap_index_to_implicant(test.num_bits, i, s);
                        LOG_WARN(
                            "test case expected implicant %s (bitmap index %d) which was not returned by "
                            "implementation",
                            s, i);
                    }
                }
            }
            bitmap_free(result.primes);
        }
    }
    for (int i = 0; i < num_testfiles; i++) {
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
    prime_implicant_result result_warmup = impl.implementation(num_bits, 0, trues);

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

    // free warmup result after measuring to prevent reuse of allocation leading to warm cache
    bitmap_free(result_warmup.primes);
    bitmap_free(result.primes);
}

void measure_merge(int num_bits) {
    LOG_INFO("measuring merge_implicants_dense bits=%d", num_bits);

    int input_elements = 1 << num_bits;
    int output_elements = num_bits << (num_bits - 1);
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