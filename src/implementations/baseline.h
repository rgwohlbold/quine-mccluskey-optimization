#pragma once

#include <stdbool.h>

#include "../implicant.h"

prime_implicant_result prime_implicants_baseline(int num_bits, int num_trues, int *trues);
void merge_implicants_baseline(bool *implicants, bool *output, bool *merged, int num_bits, int first_difference);