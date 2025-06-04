#pragma once

#include "../../implicant.h"
#include "../../util.h"
#include "../../vtune.h"
#include "../common.h"
#ifdef __x86_64__
#  include "../../tsc_x86.h"
#endif
#ifdef __aarch64__
#  include "../../kperf.h"
// #  include "../../vct_arm.h"
#endif
#include "../../my_signpost.h"

#ifndef IMPLEMENTATION_FUNCTION
#  error "need to define IMPLEMENTATION_FUNCTION"
#endif
#ifndef MERGE_FUNCTION
#  error "need to define MERGE_FUNCTION"
#endif

typedef struct {
    uint8_t  rem_bits;   // remaining bits
    uint8_t  first_diff; // first difference
    uint64_t in_idx;     // input_index
    uint64_t out_idx;    // output_index
} MergeOp;

/*
 * Compute total number of ops = sum_{d=0..num_bits} C(num_bits,d)*(num_bits-d)
 * then malloc exactly that many MergeOp slots, read them from
 * "merge_schedule_n<NUM_BITS>.txt" (whitespace: rem_bits first_diff in_idx out_idx).
 */
static MergeOp *load_schedule(int num_bits, size_t *out_count) {
    // 1) compute total
    
    size_t total = (1 << num_bits);
    

    MergeOp *ops = (MergeOp *)malloc(sizeof *ops * total);
    if (!ops) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    // 2) open file
    char fname[64];
    snprintf(fname, sizeof(fname), "traversals/flat/merge_schedule_n%d_flat.txt", num_bits);
    FILE *f = fopen(fname, "r");
    if (!f) {
        perror(fname);
        exit(EXIT_FAILURE);
    }

    // 3) read lines
    size_t idx = 0;
    while (idx < total) {
        int rem, diff;
        uint64_t in, out;
        int r = fscanf(f, "%d %d %llu %llu\n", &rem, &diff, &in, &out);
        if (r == EOF) break;
        if (r != 4) {
            // skip malformed/blank/comment lines
            char buf[256];
            if (!fgets(buf, sizeof buf, f)) break;
            continue;
        }
        ops[idx].rem_bits   = (uint8_t)rem;
        ops[idx].first_diff = (uint8_t)diff;
        ops[idx].in_idx     = in;
        ops[idx].out_idx    = out;
        idx++;
    }
    fclose(f);

    if (idx != total) {
        fprintf(stderr,
           "warning: expected %zu ops, but read only %zu\n", total, idx);
    }
    *out_count = idx;
    return ops;
}

prime_implicant_result IMPLEMENTATION_FUNCTION(int num_bits,
                                              int num_trues,
                                              int *trues)
{
    size_t num_implicants = calculate_num_implicants(num_bits);
    bitmap primes     = bitmap_allocate(num_implicants);
    bitmap implicants = bitmap_allocate(num_implicants);

    for (int i = 0; i < num_trues; i++) {
        BITMAP_SET_TRUE(implicants, trues[i]);
    }

    // load our precomputed schedule
    size_t op_count;
    MergeOp *ops = load_schedule(num_bits, &op_count);
    init_tsc();
    uint64_t counter_start = start_tsc();

    // one linear pass
    for (size_t i = 0; i < op_count; i++) {
        MergeOp *op = &ops[i];
        MERGE_FUNCTION(
            implicants,
            primes,
            op->in_idx,
            op->out_idx,
            op->rem_bits,
            op->first_diff
        );
    }

    // last implicant prime if still set
    BITMAP_SET(primes,
               num_implicants - 1,
               BITMAP_CHECK(implicants, num_implicants - 1));

    uint64_t cycles = stop_tsc(counter_start);
    free(ops);
    bitmap_free(implicants);

    prime_implicant_result result = {
        .primes = primes,
        .cycles = cycles,
    };
    return result;
}
