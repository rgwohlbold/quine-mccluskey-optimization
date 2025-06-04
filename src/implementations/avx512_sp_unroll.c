#ifdef __AVX512F__
#include "merge/avx512_sp_unroll.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_unroll
#define MERGE_FUNCTION merge_avx512_sp_unroll

#include "algorithms/sp.h"

#endif
