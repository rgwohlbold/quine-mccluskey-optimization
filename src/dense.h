#pragma once

#include <stdbool.h>

#include "implicant.h"

prime_implicant_result prime_implicants_dense(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares);
void merge_implicants_dense(bool *implicants, bool *output, bool *merged, int num_bits, int first_difference);