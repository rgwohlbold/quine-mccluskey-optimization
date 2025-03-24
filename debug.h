#include <stdarg.h>

#include "types.h"

/*
LOG_LEVEL 0: No logging
LOG_LEVEL 1: Information log (default)
LOG_LEVEL 2: Debug log
*/

#ifndef LOG_LEVEL
#define LOG_LEVEL 1
#endif
// #define LOG_NOCOLOR

#define LOG_LEVEL_DEBUG 2
#define LOG_LEVEL_INFO 1

#define _LL_INFO (LOG_LEVEL >= LOG_LEVEL_INFO)
#define _LL_DEBUG (LOG_LEVEL >= LOG_LEVEL_DEBUG)

#ifdef LOG_NOCOLOR
#define __DEBUG_PREFIX "[DEBUG]"
#define __INFO_PREFIX "[INFO ]"
#else
#define __DEBUG_PREFIX "[\033[32mDEBUG\033[39m]"
#define __INFO_PREFIX "[\033[36mINFO\033[39m ]"
#endif

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

// Convenience macros to auto-insert file and line details
#define LOG_INFO(format, ...) (_LL_INFO && _log_fmt(__INFO_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));
#define LOG_DEBUG(format, ...) (_LL_DEBUG && _log_fmt(__DEBUG_PREFIX, __BASE_FILE__, __LINE__, format, ##__VA_ARGS__));
#define LOG_INFO_IMP(imp, imp_len) (_LL_INFO && _log_imp(__INFO_PREFIX, __BASE_FILE__, __LINE__, imp, imp_len));
#define LOG_DEBUG_IMP(imp, imp_len) (_LL_DEBUG && _log_imp(__DEBUG_PREFIX, __BASE_FILE__, __LINE__, imp, imp_len));