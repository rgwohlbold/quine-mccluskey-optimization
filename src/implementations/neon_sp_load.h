#pragma once

#ifdef __aarch64__
#include "../implicant.h"

prime_implicant_result prime_implicants_neon_sp_load(int num_bits, bitmap trues);
#endif
