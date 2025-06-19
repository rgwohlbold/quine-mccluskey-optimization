#pragma once

#include <stdint.h>

typedef struct {
    int64_t l1d_cache_misses;
    int64_t l1d_cache_accesses;
} perf_result;

void perf_init();
void perf_start();
perf_result perf_stop();
