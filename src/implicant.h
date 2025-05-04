#pragma once

#include <stdio.h>
#include <stdint.h>

#include "bitmap.h"

#define COUNT_OPS 1

typedef struct {
    bitmap primes;
    uint64_t cycles;
#ifdef COUNT_OPS
    uint64_t num_ops;
#endif
} prime_implicant_result;

typedef prime_implicant_result (*implementation_function)(int num_bits, int num_trues, int *trues);

typedef struct {
    const char *name;
    implementation_function implementation;
    int max_bits;
} prime_implicant_implementation;