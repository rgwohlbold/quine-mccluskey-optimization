#pragma once

#ifdef __AVX512F__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx512_sp_old_loop(int num_bits, int num_trues, int *trues);
#endif
