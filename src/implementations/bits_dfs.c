#include <stdbool.h>
#include <stdlib.h>

#include "../util.h"
#include "common.h"
#ifdef __x86_64__
#include "../tsc_x86.h"
#include "../vtune.h"

#endif
#ifdef __aarch64__
#include "../vct_arm.h"
#endif
#include "../debug.h"
#include "bits.h"

prime_implicant_result prime_implicants_bits_dfs(int num_bits, int num_trues, int *trues) {
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes = bitmap_allocate(num_implicants);

    bitmap implicants = bitmap_allocate(num_implicants);
    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }
    bitmap merged = bitmap_allocate(num_implicants);

    uint64_t num_ops = 0;

    init_tsc();
    uint64_t counter_start = start_tsc();

    size_t *input_chunk_index = calloc((num_bits + 1), sizeof(size_t));
    size_t *output_chunk_index = calloc((num_bits + 1), sizeof(size_t));
    size_t *base_section_offset = malloc((num_bits + 1) * sizeof(size_t));
    size_t *total_chunks = malloc((num_bits + 1) * sizeof(size_t));

    output_chunk_index[0] = 1;
    base_section_offset[0] = 0;

    for (int i = 0; i <= num_bits; i++) {
        total_chunks[i] = binomial_coefficient(num_bits, i);
    }
    for (int num_dashes = 1; num_dashes <= num_bits; num_dashes++) {
        int remaining_bits = num_bits - num_dashes;
        base_section_offset[num_dashes] =
            base_section_offset[num_dashes - 1] + total_chunks[num_dashes - 1] * (1 << (remaining_bits + 1));
    }
    // // print base section offset
    // LOG_DEBUG("Base section offset:");
    // for (int i = 0; i <= num_bits; i++) {
    //     printf("Section %d: %zu\n", i, base_section_offset[i]);
    // }

    int sections_todo = num_bits + 1;
    int section_index = 0;
    size_t iterations = (1 << num_bits) - 1;
    while (iterations > 0) {
        // 1. Check if current layer needs developing.
        size_t input_chunk_idx = input_chunk_index[section_index];
        size_t output_chunk_idx = output_chunk_index[section_index];
        size_t orig_section_index = section_index;
        size_t all_chunks = total_chunks[section_index];
        bool finished = (input_chunk_idx >= all_chunks);
        bool underdeveloped = output_chunk_idx <= input_chunk_idx;
        // LOG_DEBUG("Section IN  %2zu; [inp=%d/out=%d/all=%d]", section_index, input_chunk_idx, output_chunk_idx);

        // print_bitmap_sparse("Implicants", &implicants);
        // print_bitmap_sparse_repr("Implicants", &implicants, num_bits);
        // // print_primes_sparse("Primes", &implicants, &merged);
        // 1. a) Finished: Reduce current layer. pop.

        LOG_DEBUG("Iteration: %d, Section index %d", iterations, section_index);
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
            // LOG_DEBUG("In chunk idx: %d, Out chunk idx: %d, In, out addr: %zu, %zu", input_chunk_idx, output_chunk_idx, input_index, output_index);
            int remaining_bits = num_bits - section_index;
            LOG_DEBUG("num_bits: %d, section_index: %zu, input_chunk_idx: %zu, output_chunk_idx: %zu, remaining_bits: %d",
                      num_bits, section_index, input_chunk_idx, output_chunk_idx, remaining_bits);
            int ls = leading_stars(num_bits, section_index, input_chunk_idx);
            int first_difference = remaining_bits - ls;
            
            merge_implicants_bits(implicants, merged, input_index, output_index, remaining_bits, first_difference);
            iterations--;
            input_chunk_index[section_index]++;
            output_chunk_index[section_index + 1] += ls;
            if (input_chunk_index[section_index] == total_chunks[section_index]) {
                // LOG_DEBUG("Section %2zu finished, sections_todo=%d", section_index, sections_todo);
                sections_todo--;
            }
            // 2. Go deeper
            if(last_section){
                section_index--;
            }else{
                section_index++;
            }
        }
        // LOG_DEBUG("Section OUT %2zu; [inp=%d/out=%d/all=%d]", orig_section_index, input_chunk_index[orig_section_index],
        // output_chunk_index[orig_section_index], total_chunks[orig_section_index]);
    }


    // Step 2: Scan for unmerged implicants
    for (size_t i = 0; i < num_implicants / 64; i++) {
        uint64_t implicant_true = ((uint64_t *)implicants.bits)[i];
        uint64_t merged_true = ((uint64_t *)merged.bits)[i];
        uint64_t prime_true = implicant_true & ~merged_true;
        ((uint64_t *)primes.bits)[i] = prime_true;
    }
    for (size_t i = num_implicants - (num_implicants % 64); i < num_implicants; i++) {
        if (BITMAP_CHECK(implicants, i) && !BITMAP_CHECK(merged, i)) {
            BITMAP_SET_TRUE(primes, i);
        }
    }
    uint64_t cycles = stop_tsc(counter_start);
    bitmap_free(implicants);
    bitmap_free(merged);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
#ifdef COUNT_OPS
        .num_ops = num_implicants * num_bits,
#endif
    };
    return result;
}