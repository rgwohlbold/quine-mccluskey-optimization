#pragma once

#ifdef __aarch64__
#include "../implicant.h"

prime_implicant_result prime_implicants_neon_single_pass(int num_bits, int num_trues, int *trues);
#endif
