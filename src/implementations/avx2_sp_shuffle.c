#ifdef __AVX2__
#define LOG_BLOCK_SIZE 0
#include "merge/avx2_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx2_sp_shuffle
#define MERGE_FUNCTION merge_avx2_sp_block

#include "algorithms/sp.h"

#endif
