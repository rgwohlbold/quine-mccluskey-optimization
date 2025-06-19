#ifdef __BMI2__
#define LOG_BLOCK_SIZE_PEXT 1
#include "merge/pext_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_pext_sp_load_block2
#define MERGE_FUNCTION merge_pext_sp_block

#include "algorithms/load_sp.h"
#endif
