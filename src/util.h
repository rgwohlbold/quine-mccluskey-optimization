#pragma once

#include <stdbool.h>

#include "implicant.h"

implicant allocate_minterm_array(int num_bits, int num_minterms);
bool *allocate_boolean_array(int num_elements);
void flush_cache(bool *array, int num_elements);
int calculate_num_implicants(int num_bits);