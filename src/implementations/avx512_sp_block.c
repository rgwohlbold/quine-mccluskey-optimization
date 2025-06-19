#ifdef __AVX512F__

#define LOG_BLOCK_SIZE_AVX512 4
#include "merge/avx512_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_block
#define MERGE_FUNCTION merge_avx512_sp_block

#include "algorithms/sp.h"

#endif
