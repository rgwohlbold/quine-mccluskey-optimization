#if defined(__linux__)
#define _GNU_SOURCE
#include "perf.h"
#include <linux/perf_event.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

//struct perf_event_attr hw_event;
bool initialized = false;
bool active = false;
int fd_read_misses;
int fd_read_accesses;

static int open_cache_perf(uint64_t config) {
    struct perf_event_attr pe;

    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1; // Don't count hypervisor events.

    const pid_t pid = 0;
    const int cpu = -1;
    const int group_fd = -1;
    const unsigned long flags = 0;

    int fd = syscall(__NR_perf_event_open, &pe, pid, cpu, group_fd, flags);
    if (fd == -1) {
        fprintf(stderr, "Error opening leader %llx\n", pe.config);
        exit(EXIT_FAILURE);
    }
    return fd;
}

void perf_init() {
    assert(!initialized);
    initialized = true;
    active = false;
}

void perf_start() {
    if (!initialized) {
        return;
    }
    fd_read_misses = open_cache_perf(PERF_COUNT_HW_CACHE_L1D |
                                     PERF_COUNT_HW_CACHE_OP_READ << 8 |
                                     PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    fd_read_accesses = open_cache_perf(PERF_COUNT_HW_CACHE_L1D |
                                       PERF_COUNT_HW_CACHE_OP_READ << 8 |
                                       PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    active = true;
    ioctl(fd_read_accesses, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_read_misses, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_read_accesses, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_read_misses, PERF_EVENT_IOC_ENABLE, 0);
}

perf_result perf_stop() {
    perf_result result = {
        .l1d_cache_accesses = 0,
        .l1d_cache_misses = 0,
    };
    if (!initialized) {
        return result;
    }
    assert(active);
    active = false;

    ioctl(fd_read_misses, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_read_accesses, PERF_EVENT_IOC_DISABLE, 0);
    read(fd_read_misses, &result.l1d_cache_misses, sizeof(int64_t));
    read(fd_read_accesses, &result.l1d_cache_accesses, sizeof(int64_t));
    return result;
}
#else
void perf_init() { }

void perf_start() { }

perf_result perf_stop() {
    perf_result result;
    return result;
}

#endif
