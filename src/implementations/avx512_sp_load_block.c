#ifdef __AVX512F__
#include "merge/avx512_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_load_block
#define MERGE_FUNCTION merge_avx512_sp_block

#include "algorithms/load_sp.h"

#endif
