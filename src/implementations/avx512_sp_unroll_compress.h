#pragma once

#ifdef __AVX512F__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx512_sp_unroll_compress(int num_bits, bitmap trues);
#endif
