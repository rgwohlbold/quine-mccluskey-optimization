#pragma once

#include <stdio.h>
#include <stdint.h>

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

typedef struct {
    implicant primes;
    int num_implicants;
    uint64_t cycles;
#ifdef COUNT_OPS
    uint64_t num_ops;
#endif
} prime_implicant_result;

typedef prime_implicant_result (*implementation_function)(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares);

typedef struct {
    const char *name;
    implementation_function implementation;
} prime_implicant_implementation;

void fprint_implicant(FILE *__restrict__ __stream, implicant arr, int num_bits);
void print_implicant(implicant arr, int num_bits);
int cmp_implicant(implicant a, implicant b, int num_bits);
