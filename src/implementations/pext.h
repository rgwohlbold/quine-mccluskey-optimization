#pragma once

#ifdef __BMI2__
#include "../implicant.h"

prime_implicant_result prime_implicants_pext(int num_bits, bitmap trues);
#endif
