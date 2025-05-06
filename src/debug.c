#include "debug.h"

#include <stdarg.h>

#include "implicant.h"

int _log_fmt(const char *prefix, const char *file, int line, const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("%s %s:%d: ", prefix, file, line);
    vprintf(format, args);
    printf("\n");
    va_end(args);
    return 0;
}

#ifdef __AVX2__
void log_m256i(const char *msg, const __m256i *value) {
    uint64_t buffer[4];
    _mm256_storeu_si256((__m256i*)buffer, *value);
    LOG_DEBUG("%s %lx %lx %lx %lx", msg, buffer[3], buffer[2], buffer[1], buffer[0]);
}
#endif