#pragma once

#include <stdio.h>

typedef enum {
    FV_FALSE,
    FV_TRUE,
    FV_DONT_CARE,
} function_value;

typedef enum {
    TV_FALSE,
    TV_TRUE,
    TV_DASH,
} ternary_value;

typedef ternary_value *implicant;

void fprint_implicant(FILE *__restrict__ __stream, implicant arr, int num_bits);
void print_implicant(implicant arr, int num_bits);