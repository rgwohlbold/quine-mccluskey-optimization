#ifdef VTUNE_API
#include <ittnotify.h>

#include "debug.h"
// static void init_itt_handles();

extern __itt_domain *itt_domain;
extern __itt_string_handle *itt_section_handles[32];
extern __itt_string_handle *itt_reduce_handle;

#define ITT_START_TASK_SECTION(n) (__itt_task_begin(itt_domain, __itt_null, __itt_null, itt_section_handles[n]));
#define ITT_START_GATHER_TASK() (__itt_task_begin(itt_domain, __itt_null, __itt_null, itt_reduce_handle));
#define ITT_END_TASK() (__itt_task_end(itt_domain));
#define ITT_START_FRAME() (__itt_frame_begin_v3(itt_domain, NULL));
#define ITT_END_FRAME() (__itt_frame_end_v3(itt_domain, NULL));
#else
#define ITT_START_TASK_SECTION(n)
#define ITT_START_GATHER_TASK()
#define ITT_END_TASK()
#define ITT_START_FRAME()
#define ITT_END_FRAME()
#endif
void init_itt_handles(const char* name);