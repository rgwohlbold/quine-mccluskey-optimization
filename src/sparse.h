#pragma once

#include <stdbool.h>

#include "implicant.h"

// used in tests as well
bool check_elt_in_implicant_list(int num_bits, implicant needle, ternary_value *haystack, int num_implicants);

prime_implicant_result prime_implicants_sparse(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares);