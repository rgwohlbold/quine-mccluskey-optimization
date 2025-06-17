#pragma once

#ifdef __AVX2__
#include "../implicant.h"

prime_implicant_result prime_implicants_bits_sp_load_block4(int num_bits, int num_trues, int *trues);
#endif
