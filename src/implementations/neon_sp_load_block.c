#ifdef __aarch64__
#include "merge/neon_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_neon_sp_load_block
#define MERGE_FUNCTION merge_neon_sp_block

#include "algorithms/load_sp.h"

#endif
