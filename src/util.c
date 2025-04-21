#include "util.h"

#include <stdlib.h>
#include <immintrin.h>

// calculate 3**num_bits
int calculate_num_implicants(int num_bits) {
    int num_implicants = 1;
    for (int i = 0; i < num_bits; i++) {
        num_implicants *= 3;
    }
    return num_implicants;
}

implicant allocate_minterm_array(int num_bits, int num_minterms) {
    implicant minterms = (ternary_value *)calloc(num_minterms * num_bits, sizeof(ternary_value));
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

void flush_cache(bool *array, int num_elements) {
    const int block_size = 64;

    // align pointer to 64 bytes
    uint8_t *ptr = (uint8_t *) (((unsigned long)array) & ~(block_size - 1));
    uint8_t *end = (uint8_t *) &array[num_elements];
    for (; ptr < end; ptr += block_size) {
        _mm_clflush(ptr);
    }
}