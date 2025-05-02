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