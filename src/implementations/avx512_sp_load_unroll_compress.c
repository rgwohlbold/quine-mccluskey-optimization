#ifdef __AVX512F__
#include "merge/avx512_sp_unroll_compress.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_load_unroll_compress
#define MERGE_FUNCTION merge_avx512_sp_unroll_compress

#include "algorithms/load_sp.h"

#endif
