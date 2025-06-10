#include "test.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "debug.h"
#include "implicant.h"
#ifdef __x86_64__
#include "tsc_x86.h"
#include "vtune.h"
#include "implementations/avx2.h"
#include "implementations/avx2_sp.h"
#include "implementations/avx2_sp_ssa.h"
#include "implementations/avx2_sp_ilp.h"
#include "implementations/avx2_sp_unroll.h"
#include "implementations/avx2_sp_shuffle.h"
#include "implementations/avx2_sp_load_shuffle.h"
#include "implementations/avx2_sp_load_block.h"
#include "implementations/avx512_sp_old_loop.h"
#include "implementations/avx512_sp_unroll.h"
#include "implementations/avx512_sp_unroll_compress.h"
#include "implementations/hellman.h"
#include "implementations/pext.h"

#include "implementations/merge/avx2_sp.h"
#include "implementations/merge/avx2_sp_ssa.h"
#include "implementations/merge/avx2_sp_ilp.h"
#include "implementations/merge/avx2_sp_unroll.h"
#include "implementations/merge/avx2_sp_shuffle.h"
#include "implementations/merge/avx2_sp_block.h"
#include "implementations/merge/avx512_sp_old_loop.h"
#include "implementations/merge/avx512_sp_unroll.h"
#include "implementations/merge/avx512_sp_unroll_compress.h"
#include "implementations/merge/avx2.h"
#include "implementations/merge/pext.h"



#endif
#ifdef __aarch64__
#include "vct_arm.h"
#include "implementations/merge/neon_sp.h"
#include "implementations/merge/neon.h"
#include "implementations/neon.h"
#include "implementations/neon_sp.h"
#include "implementations/neon_sp_load.h"
#include "implementations/neon_sp_dfs_load.h"
#endif

#include "implementations/baseline.h"
#include "implementations/bits.h"
#include "implementations/bits_sp.h"
#include "implementations/bits_sp_block.h"
#include "implementations/bits_sp_load.h"
#include "implementations/bits_sp_load_block.h"

#include "implementations/native_dfs_sp.h"
#include "implementations/merge/bits.h"
#include "implementations/merge/bits_sp.h"
#include "implementations/merge/bits_sp_block.h"
#include "system.h"
#include "util.h"



const prime_implicant_implementation implementations[] = {
    // {"baseline", prime_implicants_baseline, 19},
    // {"bits", prime_implicants_bits, 30},
    {"bits_sp", prime_implicants_bits_sp, 30},
    // {"bits_sp_aleksa", prime_implicants_bits_sp_aleksa, 30},
    // {"native_dfs_sp", prime_implicants_native_dfs_sp, 30},
    {"bits_sp_block", prime_implicants_bits_sp_block, 30},
    {"bits_sp_load", prime_implicants_bits_sp_load, 30},
    {"bits_sp_load_block", prime_implicants_bits_sp_load_block, 30},

#ifdef __BMI2__
    {"pext", prime_implicants_pext, 30},
#endif
#ifdef __AVX2__
    {"hellman", prime_implicants_hellman, 23},
    {"avx2", prime_implicants_avx2, 30},
    {"avx2_sp", prime_implicants_avx2_sp, 30},
    {"avx2_sp_ssa", prime_implicants_avx2_sp_ssa, 30},
    {"avx2_sp_ilp", prime_implicants_avx2_sp_ilp, 30},
    {"avx2_sp_unroll", prime_implicants_avx2_sp_unroll, 30},
    {"avx2_sp_shuffle", prime_implicants_avx2_sp_shuffle, 30},
    {"avx2_sp_load_shuffle", prime_implicants_avx2_sp_load_shuffle, 30},
    {"avx2_sp_load_block", prime_implicants_avx2_sp_load_block, 30},
#endif
#ifdef __AVX512F__
    {"avx512_sp_old_loop", prime_implicants_avx512_sp_old_loop, 22},
    {"avx512_sp_unroll", prime_implicants_avx512_sp_unroll, 22},
    {"avx512_sp_unroll_compress", prime_implicants_avx512_sp_unroll_compress, 22},
#endif
#ifdef __aarch64__
    // {"neon", prime_implicants_neon, 30},
    // {"neon_sp", prime_implicants_neon_sp, 30},
    // {"neon_sp_load", prime_implicants_neon_sp_load, 30},
    // {"neon_sp_dfs_load", prime_implicants_neon_sp_dfs_load, 30}
#endif
};

typedef struct {
    const char *name;  // static storage
    void (*impl)(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits,
                 int first_difference);
} merge_implementation;

merge_implementation merge_implementations[] = {
    {"merge_bits", merge_bits},
    {"merge_bits_sp", merge_bits_sp},
    // {"merge_bits_sp_aleksa", merge_bits_sp_aleksa},
#ifdef __BMI2__
    {"merge_pext", merge_pext},
#endif
#ifdef __AVX2__
    {"merge_avx2", merge_avx2},
    {"merge_avx2_sp", merge_avx2_sp},
    {"merge_avx2_sp_ssa", merge_avx2_sp_ssa},
    {"merge_avx2_sp_ilp", merge_avx2_sp_ilp},
    {"merge_avx2_sp_unroll", merge_avx2_sp_unroll},
    {"merge_avx2_sp_shuffle", merge_avx2_sp_shuffle},
#endif
#ifdef __AVX512F__
    {"merge_avx512_sp_old_loop", merge_avx512_sp_old_loop},
    {"merge_avx512_sp_unroll", merge_avx512_sp_unroll},
    {"merge_avx512_sp_unroll_compress", merge_avx512_sp_unroll_compress},
#endif
#ifdef __aarch64__
    {"merge_neon", merge_neon},
    {"merge_neon_sp", merge_neon_sp}
#endif
};

typedef struct {
    char *name;  // malloc'd
    int num_bits;
    int num_trues;
    int *trues;  // malloc'd
    uint64_t num_prime_implicants;
    bitmap prime_implicants;
} test_case;

void free_test(test_case test) {
    free(test.name);
    free(test.trues);
    bitmap_free(test.prime_implicants);
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
    uint64_t num_prime_implicants = 0;
    if (fgets_comments(line, sizeof(line), f) == NULL) {
        perror("could not read test file");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    line[strcspn(line, "\n")] = 0;  // remove newline
    char *name = malloc(strlen(line) + 1);
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
    sscanf(line, "%d %d %lu", &num_bits, &num_trues, &num_prime_implicants);
    LOG_DEBUG("num_bits=%d num_trues=%d num_prime_implicants=%d", num_bits, num_trues, num_prime_implicants);

    uint64_t num_implicants = calculate_num_implicants(num_bits);
    int *trues = malloc(num_trues * sizeof(int));
    bitmap prime_implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        fgets_comments(line, sizeof(line), f);
        sscanf(line, "%d", &trues[i]);
    }
    for (uint64_t i = 0; i < num_prime_implicants; i++) {
        fgets_comments(line, sizeof(line), f);
        line[strcspn(line, "\n")] = 0;  // remove newline
        uint64_t bitset_index = bitmap_implicant_to_index(num_bits, line);
        BITMAP_SET_TRUE(prime_implicants, bitset_index);
    }
    fclose(f);

    test_case result = {name, num_bits, num_trues, trues, num_prime_implicants, prime_implicants};
    *dest = result;
}

// test all implementations on one test file
void test_implementations(char **testfiles, int num_testfiles) {
    test_case test_cases[num_testfiles];
    for (int i = 0; i < num_testfiles; i++) {
        from_testfile(testfiles[i], &test_cases[i]);
    }
    for (unsigned long i = 0; i < (unsigned long)num_testfiles; i++) {
        test_case test = test_cases[i];
        for (unsigned long k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
            prime_implicant_implementation impl = implementations[k];
            if (test.num_bits > impl.max_bits) {
                LOG_INFO("skipping '%s' -> '%s'", test.name, impl.name);
                continue;
            }
            LOG_INFO("checking '%s' -> '%s'", test.name, impl.name);

            prime_implicant_result result = impl.implementation(test.num_bits, test.num_trues, test.trues);
            bool success = true;
            if (!bitmap_cmp(result.primes, test.prime_implicants)) {
                for (size_t i = 0; i < result.primes.num_bits; i++) {
                    if (BITMAP_CHECK(result.primes, i) && !BITMAP_CHECK(test.prime_implicants, i)) {
                        char s[test.num_bits + 1];
                        s[test.num_bits] = '\0';
                        bitmap_index_to_implicant(test.num_bits, i, s);
                        LOG_ERROR("returned implicant %s (bitmap index %lu) which was not expected by test case", s, i);
                        success = false;
                    }
                    if (!BITMAP_CHECK(result.primes, i) && BITMAP_CHECK(test.prime_implicants, i)) {
                        char s[test.num_bits + 1];
                        s[test.num_bits] = '\0';
                        bitmap_index_to_implicant(test.num_bits, i, s);
                        LOG_ERROR(
                            "test case expected implicant %s (bitmap index %lu) which was not returned by "
                            "implementation",
                            s, i);
                        success = false;
                    }
                }
            }
            bitmap_free(result.primes);
            if (!success) {
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < num_testfiles; i++) {
        free_test(test_cases[i]);
    }
}

// test a single implementation on a signle test file
void test_implementation_single(const char *implementation_name, char **testfiles, int num_testfiles) {
    test_case test_cases[num_testfiles];
    for (int i = 0; i < num_testfiles; i++) {
        from_testfile(testfiles[i], &test_cases[i]);
    }

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


    for (unsigned long i = 0; i < (unsigned long)num_testfiles; i++) {
        test_case test = test_cases[i];

        if (test.num_bits > impl.max_bits) {
            LOG_INFO("skipping '%s' -> '%s'", test.name, impl.name);
            continue;
        }
        LOG_INFO("checking '%s' -> '%s'", test.name, impl.name);

        prime_implicant_result result = impl.implementation(test.num_bits, test.num_trues, test.trues);
        bool ok = true;
        if (!bitmap_cmp(result.primes, test.prime_implicants)) {
            for (size_t i = 0; i < result.primes.num_bits; i++) {
                if (BITMAP_CHECK(result.primes, i) && !BITMAP_CHECK(test.prime_implicants, i)) {
                    char s[test.num_bits + 1];
                    s[test.num_bits] = '\0';
                    bitmap_index_to_implicant(test.num_bits, i, s);
                    LOG_WARN("returned implicant %s (bitmap index %d) which was not expected by test case", s, i);
                    ok = false;
                }
                if (!BITMAP_CHECK(result.primes, i) && BITMAP_CHECK(test.prime_implicants, i)) {
                    char s[test.num_bits + 1];
                    s[test.num_bits] = '\0';
                    bitmap_index_to_implicant(test.num_bits, i, s);
                    LOG_WARN(
                        "test case expected implicant %s (bitmap index %d) which was not returned by "
                        "implementation",
                        s, i);
                    ok = false;
                }
            }
        }
        if (ok)
            LOG_INFO("implementation '%s' passed test case '%s'", impl.name, test.name);
        bitmap_free(result.primes);
    }
    for (int i = 0; i < num_testfiles; i++) {
        free_test(test_cases[i]);
    }
}

void print_implementations() {
    for (unsigned long k = 0; k < sizeof(implementations) / sizeof(implementations[0]); k++) {
        printf("%s\n", implementations[k].name);
    }
}

void print_merge_implementations() {
    for (unsigned long k = 0; k < sizeof(merge_implementations) / sizeof(merge_implementations[0]); k++) {
        printf("%s\n", merge_implementations[k].name);
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
    //init_itt_handles(implementation_name);

    int trues[] = {};

    // warmup iteration
    prime_implicant_result result_warmup = impl.implementation(num_bits, 0, trues);

    LOG_INFO("measuring '%s' bits=%d", impl.name, num_bits);
    // ITT_START_FRAME();
    prime_implicant_result result = impl.implementation(num_bits, 0, trues);
    // ITT_END_FRAME();
    uint64_t cycles = result.cycles;
    FILE *f = fopen("measurements.csv", "a");
    fprintf(f, "%s,%s,%s,%s,%d,%lu\n", compiler_version, compiler_flags, cpu_model, impl.name, num_bits, cycles);
    fclose(f);

    // free warmup result after measuring to prevent reuse of allocation leading to warm cache
    bitmap_free(result_warmup.primes);
    bitmap_free(result.primes);
}

void measure_merge(const char *s, int num_bits) {
    merge_implementation impl;
    bool implementation_found = false;
    for (unsigned long k = 0; k < sizeof(merge_implementations) / sizeof(merge_implementations[0]); k++) {
        impl = merge_implementations[k];
        if (strcmp(impl.name, s) == 0) {
            implementation_found = true;
            break;
        }
    }
    if (!implementation_found) {
        LOG_WARN("could not find implementation %s", s);
        return;
    }
    LOG_INFO("measuring %s bits=%d", impl.name, num_bits);

    size_t input_elements = 1 << num_bits;
    size_t output_elements = num_bits << (num_bits - 1);
    const int warmup_iterations = 10;
    bitmap implicants_warmup = bitmap_allocate(input_elements + output_elements);
    bitmap merged_warmup = bitmap_allocate(input_elements);

    for (int i = 0; i < warmup_iterations; i++) {
        // use first difference 0 for now
        impl.impl(implicants_warmup, merged_warmup, 0, input_elements, num_bits, 0);
    }

    bitmap implicants = bitmap_allocate(input_elements + output_elements);
    bitmap merged = bitmap_allocate(input_elements);
    // TODO: check if this makes a difference
    flush_cache(implicants.bits, (input_elements + output_elements + 7) / 8);
    flush_cache(merged.bits, (input_elements + 7) / 8);

    init_tsc();
    uint64_t counter = start_tsc();
    impl.impl(implicants, merged, 0, input_elements, num_bits, 0);
    uint64_t cycles = stop_tsc(counter);

    bitmap_free(implicants_warmup);
    bitmap_free(merged_warmup);
    bitmap_free(implicants);
    bitmap_free(merged);

    FILE *f = fopen("measurements_merge.csv", "a");
    fprintf(f, "%s,%s,%s,%s,%d,%lu\n", compiler_version, compiler_flags, cpu_model, impl.name, num_bits, cycles);
    fclose(f);
}

void generate_testfile(int num_bits, int density) {
    // allocate full 2**n ints in case we are a bit above density
    int max_implicants = 1 << num_bits;
    int *trues = calloc(max_implicants, sizeof(int));
    if (trues == NULL) {
        LOG_ERROR("could not allocate trues array");
        exit(EXIT_FAILURE);
    }

    int num_trues = 0;
    srand(time(NULL));
    for (int i = 0; i < max_implicants; i++) {
        double ratio = ((double)rand()) / RAND_MAX * 100;
        if (ratio < density) {
            trues[num_trues++] = i;
        }
    }

    prime_implicant_result result = prime_implicants_bits(num_bits, num_trues, trues);
    bitmap primes = result.primes;

    char filename[100];
    snprintf(filename, sizeof(filename), "tests/gen_tests/rnd-%d-%d.txt", num_bits, density);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        perror("could not open test file");
        exit(EXIT_FAILURE);
    }
    size_t num_primes = 0;
    for (size_t i = 0; i < primes.num_bits; i++) {
        if (BITMAP_CHECK(primes, i)) {
            num_primes++;
        }
    }
    fprintf(f, "rnd-%d-%dpct-dense\n", num_bits, density);
    fprintf(f, "%d %d %lu\n", num_bits, num_trues, num_primes);
    for (int i = 0; i < num_trues; i++) {
        fprintf(f, "%d\n", trues[i]);
    }
    for (size_t i = 0; i < primes.num_bits; i++) {
        if (BITMAP_CHECK(primes, i)) {
            char s[num_bits+1];
            s[num_bits] = '\0';
            bitmap_index_to_implicant(num_bits, i, s);
            fprintf(f, "%s\n", s);
        }
    }
    fclose(f);
}
