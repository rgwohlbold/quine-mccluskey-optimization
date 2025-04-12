#pragma once

#include "implicant.h"

prime_implicant_result prime_implicants_dense(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares);