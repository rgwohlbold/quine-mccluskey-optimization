#pragma once

#include <stdbool.h>
#include <assert.h>

#include "implicant.h"

bool *allocate_boolean_array(int num_elements);
void flush_cache(bool *array, int num_elements);
int calculate_num_implicants(int num_bits);

extern const int binomial_coefficients[20][20];

static inline int binomial_coefficient(int n, int k) {
    assert(0 <= n);
    assert(0 <= k);
    assert(n < 20);
    assert(k < 20);
    return binomial_coefficients[n][k];
}