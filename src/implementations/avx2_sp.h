#pragma once

#ifdef __AVX2__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx2_sp(int num_bits, bitmap trues);
#endif
