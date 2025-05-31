#pragma once

#include "implicant.h"

void print_implementations();
void print_merge_implementations();
void test_implementations(char **testfiles, int num_testfiles);
void test_implementation_single(const char *implementation, char **testfiles, int num_testfiles);
void measure_implementations(const char *implementation, int num_bits);
void measure_merge(const char *implementation, int num_bits);
