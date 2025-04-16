#pragma once

#include "implicant.h"
#include "dense.h"
#include "sparse.h"

void test_implementations();
void measure_implementations(const char *implementation, int num_bits);
void measure_merge(int num_bits);