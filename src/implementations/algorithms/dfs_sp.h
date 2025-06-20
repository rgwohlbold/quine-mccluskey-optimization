#pragma once
#include "../../debug.h"
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

#ifndef IMPLEMENTATION_FUNCTION
#error "need to define IMPLEMENTATION_FUNCTION"
#endif

#ifndef MERGE_FUNCTION
#error "need to define MERGE_FUNCTION"
#endif

prime_implicant_result IMPLEMENTATION_FUNCTION(int num_bits, bitmap trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    // OR the trues into the implicants
    size_t num_minterms = 1 << num_bits;
    for (size_t i = 0; i < num_minterms; i++) {
        BITMAP_SET(implicants, i, BITMAP_CHECK(trues, i));
    }

    init_tsc();
    perf_start();
    uint64_t counter_start = start_tsc();

    size_t *input_chunk_index = calloc((num_bits + 1), sizeof(size_t));
    size_t *output_chunk_index = calloc((num_bits + 1), sizeof(size_t));
    size_t *base_section_offset = malloc((num_bits + 1) * sizeof(size_t));
    size_t *total_chunks = malloc((num_bits + 1) * sizeof(size_t));

    output_chunk_index[0] = 1;

    for (int i = 0; i <= num_bits; i++) {
        total_chunks[i] = binomial_coefficient(num_bits, i);
    }
    base_section_offset[0] = 0;
    for (int num_dashes = 1; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        base_section_offset[num_dashes] =
            base_section_offset[num_dashes - 1] + total_chunks[num_dashes - 1] * (1 << (remaining_bits + 1));
    }

    int section_index = 0;

    while (section_index >= 0) {
        // ITT_START_TASK_SECTION(section_index);

        // 1. Check if current layer needs developing.
        size_t layer_input_idx = input_chunk_index[section_index];
        size_t layer_output_idx = output_chunk_index[section_index];
        size_t layer_total_chunks = total_chunks[section_index];

        // LOG_DEBUG("Section IN  %2zu; [inp=%d/out=%d/all=%d]", section_index, layer_input_idx, layer_output_idx,
        //           layer_total_chunks);

        bool finished = (layer_input_idx >= layer_total_chunks);
        bool underdeveloped = layer_output_idx <= layer_input_idx;

        // 1. a) Finished: Reduce current layer. pop.
        if (num_bits == section_index || underdeveloped || finished) {
            // LOG_DEBUG("undeveloped");
            section_index--;
        }
        // 1. b) Not finished: Develop current layer.
        else {
            // LOG_DEBUG("Developing chunk S=%zu/C=%zu", section_index, layer_input_idx);

            size_t input_index =
                base_section_offset[section_index] + (layer_input_idx * (1 << (num_bits - section_index)));
            size_t output_index = base_section_offset[section_index + 1] +
                                  (output_chunk_index[section_index + 1] * (1 << (num_bits - section_index - 1)));
            int remaining_bits = num_bits - section_index;
            int leading_value = leading_stars(num_bits, section_index, layer_input_idx);
            int first_difference = remaining_bits - leading_value;
            LOG_DEBUG("Section %2zu; [inp=%zu/out=%zu/rem=%d/first_diff=%d]", section_index, input_index, output_index,
                      remaining_bits, first_difference);
            MERGE_FUNCTION(implicants, primes, input_index, output_index, remaining_bits, first_difference);

            input_chunk_index[section_index]++;
            output_chunk_index[section_index + 1] += leading_value;

            /*
             * if (input_chunk_index[section_index] == total_chunks[section_index]) {
             *     LOG_DEBUG("Section %2zu finished.", section_index);
             * }
             * */

            bool last_section = (section_index == num_bits);
            if (last_section) {
                section_index--;
            } else {
                section_index++;
            }
        }
        // ITT_END_TASK();
    }

    // mark last implicant prime if it is true
    BITMAP_SET(primes, num_implicants - 1, BITMAP_CHECK(implicants, num_implicants - 1));

    uint64_t cycles = stop_tsc(counter_start);
    perf_result pres = perf_stop();
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
        .l1d_cache_misses = pres.l1d_cache_misses,
        .l1d_cache_accesses = pres.l1d_cache_accesses,
    };
    return result;
}
