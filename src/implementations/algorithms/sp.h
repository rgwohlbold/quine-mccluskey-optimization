#pragma once

#include "../../implicant.h"
#include "../../util.h"
#include "../../vtune.h"
#include "../common.h"
#ifdef __x86_64__
#include "../../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../../vct_arm.h"
#endif
#include "../../signpost.h"

#ifndef IMPLEMENTATION_FUNCTION
#error "need to define IMPLEMENTATION_FUNCTION"
#endif

#ifndef MERGE_FUNCTION
#error "need to define MERGE_FUNCTION"
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

prime_implicant_result IMPLEMENTATION_FUNCTION(int num_bits, bitmap trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    // OR the trues into the implicants
    int num_minterms = 1 << num_bits;
    for (size_t i = 0; i < num_minterms; i++) {
        BITMAP_SET(implicants, i, BITMAP_CHECK(trues, i));
    }

    init_itt_handles(XSTR(IMPLEMENTATION_FUNCTION));
    init_tsc();
    ITT_START_FRAME()
    uint64_t counter_start = start_tsc();

    size_t input_index = 0;
    for (int num_dashes = 0; num_dashes < num_bits; num_dashes++) {
        ITT_START_TASK_SECTION(num_dashes);
        int remaining_bits = num_bits - num_dashes;
        int iterations = binomial_coefficient(num_bits, num_dashes);
        int input_elements = 1 << remaining_bits;
        int output_elements = 1 << (remaining_bits - 1);

        size_t output_index = input_index + iterations * input_elements;
        for (int i = 0; i < iterations; i++) {
            int first_difference = remaining_bits - leading_stars(num_bits, num_dashes, i);
            MERGE_FUNCTION(implicants, primes, input_index, output_index, remaining_bits, first_difference);
            output_index += (remaining_bits - first_difference) * output_elements;
            input_index += input_elements;
        }
        ITT_END_TASK();
    }
    // mark last implicant prime if it is true
    BITMAP_SET(primes, num_implicants - 1, BITMAP_CHECK(implicants, num_implicants - 1));

    uint64_t cycles = stop_tsc(counter_start);
    ITT_END_FRAME()
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
    };
    return result;
}
