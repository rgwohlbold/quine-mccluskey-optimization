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
#include "../../perf.h"
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
    size_t num_minterms = 1 << num_bits;
    for (size_t i = 0; i < num_minterms; i++) {
        BITMAP_SET(implicants, i, BITMAP_CHECK(trues, i));
    }

    init_itt_handles(XSTR(IMPLEMENTATION_FUNCTION));
    init_tsc();
    perf_start();
    ITT_START_FRAME()
    uint64_t counter_start = start_tsc();

    uint64_t input_index = 0;
    for (int num_dashes = 0; num_dashes < num_bits; num_dashes++) {
        ITT_START_TASK_SECTION(num_dashes);
        uint64_t remaining_bits = num_bits - num_dashes;
        uint64_t iterations = binomial_coefficient(num_bits, num_dashes);
        uint64_t input_elements = 1 << remaining_bits;
        uint64_t output_elements = 1 << (remaining_bits - 1);

        uint64_t output_index = input_index + iterations * input_elements;
        for (uint64_t i = 0; i < iterations; i++) {
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
    perf_result pres = perf_stop();
    ITT_END_FRAME()
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
        .l1d_cache_misses = pres.l1d_cache_misses,
        .l1d_cache_accesses = pres.l1d_cache_accesses,
    };
    return result;
}
