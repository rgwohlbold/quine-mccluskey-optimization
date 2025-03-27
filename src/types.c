#include <stdio.h>
#include "types.h"

void fprint_implicant(FILE *__restrict__ __stream, implicant arr, int num_bits) {
    for (int k = 0; k < num_bits; k++) {
        ternary_value val = arr[k];
        switch (val) {
            case TV_TRUE:
                fprintf(__stream, "1");
                break;
            case TV_FALSE:
                fprintf(__stream, "0");
                break;
            case TV_DASH:
                fprintf(__stream, "-");
                break;
        }
    }
    fprintf(__stream, "\n");
}

void print_implicant(implicant arr, int num_bits) { fprint_implicant(stdout, arr, num_bits); }