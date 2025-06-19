#pragma once

#include <stdio.h>
#include <stdint.h>

#include "bitmap.h"

typedef struct {
    bitmap primes;
    uint64_t cycles;
} prime_implicant_result;

typedef prime_implicant_result (*implementation_function)(int num_bits, bitmap trues);

typedef struct {
    const char *name;
    implementation_function implementation;
    int max_bits;
} prime_implicant_implementation;
