#ifdef __AVX2__
#define LOG_BLOCK_SIZE 1
#include "merge/bits_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_bits_sp_load_block2
#define MERGE_FUNCTION merge_bits_sp_block

#include "algorithms/load_sp.h"

#endif
