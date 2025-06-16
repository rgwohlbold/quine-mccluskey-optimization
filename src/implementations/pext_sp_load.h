#pragma once

#ifdef __BMI2__
#include "../implicant.h"

prime_implicant_result prime_implicants_pext_sp_load(int num_bits, int num_trues, int *trues);
#endif
