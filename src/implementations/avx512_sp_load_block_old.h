#pragma once

#ifdef __AVX512F__
#include "../implicant.h"

prime_implicant_result prime_implicants_avx512_sp_load_block_old(int num_bits, int num_trues, int *trues);
#endif
