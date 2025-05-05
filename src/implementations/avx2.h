#pragma once

#ifdef __AVX2__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx2(int num_bits, int num_trues, int *trues);
void merge_implicants_avx2(bitmap implicants, bitmap merged, size_t input_index, size_t output_index, int num_bits, int first_difference);
#endif