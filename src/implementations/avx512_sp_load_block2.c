#ifdef __AVX512F__

#define LOG_BLOCK_SIZE_AVX512 1
#include "merge/avx512_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_load_block2
#define MERGE_FUNCTION merge_avx512_sp_block

#include "algorithms/load_sp.h"

#endif
