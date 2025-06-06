#ifdef __AVX2__
#include "merge/avx2_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx2_sp_load_block
#define MERGE_FUNCTION merge_avx2_sp_block

#include "algorithms/load_sp.h"

#endif
