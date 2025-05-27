#include <stdbool.h>
#include <stdlib.h>

#include "../util.h"
#include "common.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../debug.h"
#include "../vtune.h"
#include "avx2_single_pass.h"

prime_implicant_result prime_implicants_avx2sp_dfs(int num_bits, int num_trues, int *trues) {
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

    int sections_todo = num_bits + 1;
    int section_index = 0;
    while (sections_todo > 0) {
        ITT_START_TASK_SECTION(section_index);
        // 1. Check if current layer needs developing.
        size_t input_chunk_idx = input_chunk_index[section_index];
        size_t output_chunk_idx = output_chunk_index[section_index];
        size_t orig_section_index = section_index;
        size_t all_chunks = total_chunks[section_index];
        bool finished = (input_chunk_idx >= all_chunks);
        bool underdeveloped = output_chunk_idx <= input_chunk_idx;
        LOG_DEBUG("Section IN  %2zu; [inp=%d/out=%d/all=%d]", section_index, input_chunk_idx, output_chunk_idx,
                  all_chunks);

        // print_bitmap_sparse("Implicants", &implicants);
        // print_bitmap_sparse("Merged", &primes);

        // 1. a) Finished: Reduce current layer. pop.
        if (finished || underdeveloped) {
            section_index--;
        }
        // 1. b) Not finished: Develop current layer.
        else {
            bool last_section = (section_index == num_bits);

            // LOG_DEBUG("Section %2zu: Developing", section_index);
            size_t input_index = base_section_offset[section_index] +
                                 input_chunk_index[section_index] * (1 << (num_bits - section_index));

            size_t output_index = base_section_offset[section_index + 1] +
                                  output_chunk_index[section_index + 1] * (1 << (num_bits - section_index - 1));

            int remaining_bits = num_bits - section_index;
            int ls = leading_stars(num_bits, section_index, input_chunk_idx);
            int first_difference = remaining_bits - ls;

            merge_implicants_avx2_single_pass(implicants, primes, input_index, output_index, remaining_bits,
                                              first_difference);

            input_chunk_index[section_index]++;
            output_chunk_index[section_index + 1] += ls;
            if (input_chunk_index[section_index] == total_chunks[section_index]) {
                LOG_DEBUG("Section %2zu finished, sections_todo=%d", section_index, sections_todo);
                sections_todo--;
            }
            // 2. Go deeper
            if (last_section) {
                section_index--;
            } else {
                section_index++;
            }
        }
        LOG_DEBUG("Section OUT %2zu; [inp=%d/out=%d/all=%d]", orig_section_index, input_chunk_index[orig_section_index],
                  output_chunk_index[orig_section_index], total_chunks[orig_section_index]);
        ITT_END_TASK();
    }

    BITMAP_SET(primes, num_implicants - 1, BITMAP_CHECK(implicants, num_implicants - 1));

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