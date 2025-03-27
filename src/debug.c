#include <stdarg.h>

#include "types.h"
#include "debug.h"

int _log_fmt(const char *prefix, const char *file, int line, const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("%s %s:%d: ", prefix, file, line);
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

int _log_imp(const char *prefix, const char *file, int line, implicant imp, int length) {
    printf("%s %s:%d: ", prefix, file, line);
    print_implicant(imp, length);
}
