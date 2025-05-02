#pragma once

#include <stdbool.h>

#include "implicant.h"

bool *allocate_boolean_array(int num_elements);
void flush_cache(bool *array, int num_elements);
int calculate_num_implicants(int num_bits);
int binomial_coefficient(int n, int k);