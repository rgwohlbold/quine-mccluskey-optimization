#include <stdio.h>
#include <stdlib.h>

typedef enum {
    FV_FALSE,
    FV_TRUE,
    FV_DONT_CARE,
} function_value;

/**
 * Compute prime implicants of the specified function
 * 
 * values is a pointer to an array of 2**num_bits function values.
 * TODO: specify return value
 */
void prime_implicants(int num_bits, function_value *values) {
    // TODO: implement 
}

function_value *build_table(int num_bits, int num_trues, int *trues, int num_dont_cares, int *dont_cares) {
    int num_values = 1 << num_bits;
    function_value *table = calloc(num_values, sizeof(function_value));
    if (table == NULL) {
        perror("could not allocate function value table");
        exit(EXIT_FAILURE);
    }
    /* this could be removed due to calloc() zero-initializing memory,
    but leaving it in for completeness' sake */
    for (int i = 0; i < num_values; i++) {
        table[i] = FV_FALSE;
    }
    for (int i = 0; i < num_trues; i++) {
        table[trues[i]] = FV_TRUE;
    }
    for (int i = 0; i < num_dont_cares; i++) {
        table[dont_cares[i]] = FV_DONT_CARE;
    }
    return table;
}

void print_table(int num_bits, function_value *table) {
    int num_values = 1 << num_bits;
    for (int i = 0; i < num_values; i++) {
        switch (table[i]) {
            case FV_TRUE:
                printf("%d\t1\n", i);
                break;
            case FV_FALSE:
                printf("%d\t0\n", i);
                break;
            case FV_DONT_CARE:
                printf("%d\tX\n", i);
                break;
        }
    }
}

int main(int argc, char **argv) {
    int num_bits = 4;
    int trues[] = {4, 8, 10, 11, 12, 15};
    int dont_cares[] = {9, 14};
    function_value *example_function = build_table(4, sizeof(trues) / sizeof(trues[0]), trues, sizeof(dont_cares) / sizeof(dont_cares[0]), dont_cares);
    print_table(num_bits, example_function);
    free(example_function);
}