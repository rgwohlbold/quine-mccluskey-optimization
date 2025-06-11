#ifdef __AVX2__
#include "merge/avx2_sp_unroll.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx2_sp_load_unroll
#define MERGE_FUNCTION merge_avx2_sp_unroll

#include "algorithms/load_sp.h"

#endif
