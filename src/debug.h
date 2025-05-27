#pragma once
#include <stdarg.h>

#include "implicant.h"

/*
LOG_LEVEL 0: No logging
LOG_LEVEL 1: Information log (default)
LOG_LEVEL 2: Debug log
*/

#ifndef LOG_LEVEL
#define LOG_LEVEL 2
#endif
// #define LOG_NOCOLOR

#define LOG_LEVEL_DEBUG 3
#define LOG_LEVEL_INFO 2
#define LOG_LEVEL_WARN 1

#define _LL_INFO (LOG_LEVEL >= LOG_LEVEL_INFO)
#define _LL_DEBUG (LOG_LEVEL >= LOG_LEVEL_DEBUG)
#define _LL_WARN (LOG_LEVEL >= LOG_LEVEL_WARN)

#ifdef LOG_NOCOLOR
#define __DEBUG_PREFIX "[DEBUG]"
#define __INFO_PREFIX "[INFO ]"
#define __WARN_PREFIX "[WARN ]"
#define __ERROR_PREFIX "[ERROR]"
#else
#define __DEBUG_PREFIX "[\033[32mDEBUG\033[39m]"
#define __INFO_PREFIX "[\033[36mINFO\033[39m ]"
#define __WARN_PREFIX "[\033[33mWARN\033[39m]"
#define __ERROR_PREFIX "[\033[31mERROR\033[39m]"
#endif

int _log_fmt(const char *prefix, const char *file, int line, const char *format, ...);

// Convenience macros to auto-insert file and line details
#define LOG_INFO(format, ...) \
    (void)(_LL_INFO && _log_fmt(__INFO_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));
#define LOG_DEBUG(format, ...) \
    (void)(_LL_DEBUG && _log_fmt(__DEBUG_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));
#define LOG_WARN(format, ...) \
    (void)(_LL_WARN && _log_fmt(__WARN_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));
#define LOG_ERROR(format, ...) (void)(_log_fmt(__ERROR_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));

void print_bitmap(const bitmap *map);
void print_bitmap_sparse(const char *msg, const bitmap *map);
void print_bitmap_bits(const bitmap *map, size_t n);
void print_primes_sparse(const char *msg, const bitmap* implicants, const bitmap* merged);
void print_bitmap_sparse_repr(const bitmap *map, int num_bits);

#ifdef __AVX2__
#include <x86intrin.h>
void log_m256i(const char *msg, const __m256i *value);
#endif

#ifdef __aarch64__
#include <arm_neon.h>
void log_u64x2(const char *msg, const uint64x2_t *value);
void log_u64x2_binary(const char *msg, const uint64x2_t *value);
#endif

