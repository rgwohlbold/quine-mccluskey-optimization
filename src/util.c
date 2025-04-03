#include "util.h"

#include <stdlib.h>

implicant allocate_minterm_array(int num_bits) {
    implicant minterms = (ternary_value *)calloc((1 << num_bits) * num_bits, sizeof(ternary_value));
    if (minterms == NULL) {
        perror("could not allocate minterms array");
        exit(EXIT_FAILURE);
    }
    return minterms;
}

bool *allocate_boolean_array(int num_elements) {
    bool *arr = (bool *)calloc(num_elements, sizeof(bool));
    if (arr == NULL) {
        perror("could not allocate boolean array");
        exit(EXIT_FAILURE);
    }
    return arr;
}
