#ifdef __AVX512F__
#include "merge/avx512_sp_block_old.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_load_block_old
#define MERGE_FUNCTION merge_avx512_sp_block_old

#include "algorithms/load_sp.h"

#endif
