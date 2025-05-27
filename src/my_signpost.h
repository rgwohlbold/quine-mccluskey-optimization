// my_signpost.h
#pragma once

#ifdef __aarch64__
#ifdef APPLE_PROFILING
#include <os/log.h>
#include <os/signpost.h>

// Create one global log and signpost-id for your module
static os_log_t          gLog;
static os_signpost_id_t  gSpid;
static inline void init_signpost(void) {
    static bool inited = false;
    if (!inited) {
        // “prime_implicants” is your subsystem; pick any identifier you like
        gLog  = os_log_create("com.myorg.prime", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
        gSpid = os_signpost_id_generate(gLog);
        inited = true;
    }
}
#define SIGNPOST_INIT() init_signpost()
// Macros for profiling
#define SIGNPOST_INTERVAL_BEGIN(gLog, gSpid, name, format, ...) \
    os_signpost_interval_begin(gLog, gSpid, name, format, ##__VA_ARGS__)

#define SIGNPOST_INTERVAL_END(gLog, gSpid, name, format, ...) \
    os_signpost_interval_end(gLog, gSpid, name, format, ##__VA_ARGS__)

#define SIGNPOST_EVENT(gLog, gSpid, name, format, ...) \
    os_signpost_event_emit(gLog, gSpid, name, format, ##__VA_ARGS__)
#endif
#else
// If PROFILING is not defined, make the macros no-ops
#define SIGNPOST_INIT() ((void)0)
#define SIGNPOST_INTERVAL_BEGIN(gLog, gSpid, name, format, ...) ((void)0)
#define SIGNPOST_INTERVAL_END(gLog, gSpid, name, format, ...) ((void)0)
#define SIGNPOST_EVENT(gLog, gSpid, name, format, ...) ((void)0)
#endif