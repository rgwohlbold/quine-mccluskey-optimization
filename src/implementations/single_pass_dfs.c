#include <stdbool.h>
#include <stdlib.h>

#include "../debug.h"
#include "../util.h"
#include "../vtune.h"
#include "bits.h"
#include "bits_single_pass.h"
#include "common.h"

#ifdef __x86_64__
#include "../tsc_x86.h"
#include "merge/avx2_sp.h"
#endif

#ifdef __aarch64__
#include "../vct_arm.h"
#include "neon_single_pass.h"
#endif

prime_implicant_result prime_implicants_single_pass_dfs(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }

    uint64_t num_ops = 0;

    init_tsc();
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

    size_t iterations = (1 << num_bits) - 1;
    while (section_index >= 0) {
        ITT_START_TASK_SECTION(section_index);

        // 1. Check if current layer needs developing.
        size_t layer_input_idx = input_chunk_index[section_index];
        size_t layer_output_idx = output_chunk_index[section_index];
        size_t layer_total_chunks = total_chunks[section_index];

        LOG_DEBUG("Section IN  %2zu; [inp=%d/out=%d/all=%d]", section_index, layer_input_idx, layer_output_idx,
                  layer_total_chunks);

        bool finished = (layer_input_idx >= layer_total_chunks);
        bool underdeveloped = layer_output_idx <= layer_input_idx;

        // 1. a) Finished: Reduce current layer. pop.
        if (underdeveloped || finished) {
            section_index--;
        }
        // 1. b) Not finished: Develop current layer.
        else {
            LOG_DEBUG("Developing chunk S=%zu/C=%zu", section_index, layer_input_idx);

            size_t input_index =
                base_section_offset[section_index] + (layer_input_idx * (1 << (num_bits - section_index)));

            size_t output_index = base_section_offset[section_index + 1] +
                                  (output_chunk_index[section_index + 1] * (1 << (num_bits - section_index - 1)));

            int remaining_bits = num_bits - section_index;
            int leading_value = leading_stars(num_bits, section_index, layer_input_idx);
            int first_difference = remaining_bits - leading_value;

#if defined(__AVX2__)
            merge_avx2_sp(implicants, primes, input_index, output_index, remaining_bits,
                                              first_difference);
#elif defined(__aarch64__)
            merge_implicants_neon_single_pass(implicants, primes, input_index, output_index, remaining_bits,
                                              first_difference);
#else
            merge_implicants_bits_single_pass(implicants, primes, input_index, output_index, remaining_bits,
                                              first_difference);
#endif

#ifdef COUNT_OPS
            num_ops += 3 * remaining_bits * (1 << (remaining_bits - 1));
#endif
            iterations--;
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
        ITT_END_TASK();
    }

    // mark last implicant prime if it is true
    BITMAP_SET(primes, num_implicants - 1, BITMAP_CHECK(implicants, num_implicants - 1));

#ifdef COUNT_OPS
    num_ops += 2 * num_implicants;
#endif

    uint64_t cycles = stop_tsc(counter_start);
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
#ifdef COUNT_OPS
        .num_ops = num_ops,
#endif
    };
    return result;
}
