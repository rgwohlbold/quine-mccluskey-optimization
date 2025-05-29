#ifdef __aarch64__
#include "merge/neon_sp.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_neon_sp_dfs_load
#define MERGE_FUNCTION merge_neon_sp

#include "algorithms/load_sp_dfs.h"

#endif
