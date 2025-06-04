#ifdef __AVX512F__
#include "merge/avx512_sp_unroll_compress.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_unroll_compress
#define MERGE_FUNCTION merge_avx512_sp_unroll_compress

#include "algorithms/sp.h"

#endif
