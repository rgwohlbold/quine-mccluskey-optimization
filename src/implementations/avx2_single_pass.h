#pragma once

#ifdef __AVX2__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx2_single_pass(int num_bits, int num_trues, int *trues);
#endif
