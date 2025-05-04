#pragma once

#include <assert.h>
#include "../util.h"

static inline int leading_stars(int num_bits, int num_dashes, int chunk_index) {
    int dashes_passed = 0;
    for (int i = 0; i < num_bits; i++) {
        int dashes_remaining = num_dashes - dashes_passed;
        if (dashes_remaining == 0) {
            return num_bits - i;
        } else {
            int possibilities_if_dash_is_set = binomial_coefficient(num_bits-i-1, dashes_remaining-1);
            if (chunk_index < possibilities_if_dash_is_set) {
                dashes_passed++;
            } else {
                chunk_index -= possibilities_if_dash_is_set;
            }
        }
    }
    return 0;
}
