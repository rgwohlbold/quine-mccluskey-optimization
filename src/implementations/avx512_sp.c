#ifdef __AVX512F__
#include "merge/avx512_sp.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_avx512_sp
#define MERGE_FUNCTION merge_avx512_sp

#include "algorithms/sp.h"

#endif
