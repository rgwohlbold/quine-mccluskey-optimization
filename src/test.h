#pragma once

#include "implicant.h"
#include "dense.h"

void test_implementations(char **testfiles, int num_testfiles);
void measure_implementations(const char *implementation, int num_bits);
void measure_merge(int num_bits);