#ifdef __AVX512F__
#include "merge/avx512_sp_old_loop.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp_old_loop
#define MERGE_FUNCTION merge_avx512_sp_old_loop

#include "algorithms/sp.h"

#endif
