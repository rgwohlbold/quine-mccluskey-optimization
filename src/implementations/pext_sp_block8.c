#ifdef __BMI2__
#define LOG_BLOCK_SIZE_PEXT 3
#include "merge/pext_sp_block.h"

#define IMPLEMENTATION_FUNCTION prime_implicants_pext_sp_block8
#define MERGE_FUNCTION merge_pext_sp_block

#include "algorithms/sp.h"
#endif
