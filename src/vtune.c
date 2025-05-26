#ifdef VTUNE_API
#include "vtune.h"

#include <ittnotify.h>

#include "debug.h"
// static void init_itt_handles();

__itt_domain *itt_domain = NULL;
__itt_string_handle *itt_section_handles[32];
__itt_string_handle *itt_reduce_handle;
void init_itt_handles() {
    if (itt_domain != NULL) {
        return;  // already initialized
    }
    LOG_INFO("initializing VTune handles");
    itt_domain = __itt_domain_create("prime_implicant");
    itt_reduce_handle = __itt_string_handle_create("prime_implicant.reduce");
    for (int i = 0; i < 32; i++) {
        char use_name[64];
        snprintf(use_name, sizeof(use_name), "prime_implicant.num_bits=%02d", i);
        itt_section_handles[i] = __itt_string_handle_create(use_name);
    }
}
#endif