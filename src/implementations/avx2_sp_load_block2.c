#ifdef __AVX2__
#define LOG_BLOCK_SIZE_AVX2 1
#include "merge/avx2_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx2_sp_load_block2
#define MERGE_FUNCTION merge_avx2_sp_block

#include "algorithms/load_sp.h"

#endif
