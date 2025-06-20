#pragma once

#include <stdbool.h>
#include <assert.h>

#include "implicant.h"

bool *allocate_boolean_array(int num_elements);
void flush_cache(uint8_t *array, int num_elements);
size_t calculate_num_implicants(int num_bits);

extern const uint64_t binomial_coefficients[30][30];

static inline uint64_t binomial_coefficient(int n, int k) {
    return binomial_coefficients[n][k];
}
