#pragma once

#ifdef __aarch64__
#include <arm_neon.h>
#include "../implicant.h"

prime_implicant_result prime_implicants_neon_single_pass(int num_bits, int num_trues, int *trues);
void merge_implicants_neon_single_pass(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference);
#endif

