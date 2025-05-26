#ifdef VTUNE_API
#include "vtune.h"

#include <ittnotify.h>

#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// static void init_itt_handles();

__itt_domain *itt_domain = NULL;
__itt_string_handle *itt_section_handles[32];
__itt_string_handle *itt_reduce_handle;
void init_itt_handles(const char* implementation_name) {
    if (itt_domain != NULL) {
        return;  // already initialized
    }
    LOG_INFO("initializing VTune handles");
    char domain_name[128];
    snprintf(domain_name, sizeof(domain_name), "impl.%s", implementation_name);
    itt_domain = __itt_domain_create(domain_name);
    snprintf(domain_name, sizeof(domain_name), "impl.%s.reduce", implementation_name);
    itt_reduce_handle = __itt_string_handle_create(domain_name);
    for (int i = 0; i < 32; i++) {
        char use_name[128];
        snprintf(use_name, sizeof(use_name), "impl.%s.num_bits=%02d", implementation_name, i);
        itt_section_handles[i] = __itt_string_handle_create(use_name);
    }
}
#endif