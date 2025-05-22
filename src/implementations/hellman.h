#pragma once

#ifdef __cplusplus

extern "C" { 
    
#endif

    #include "../implicant.h"
    #include "../bitmap.h"
    #include "../util.h"

    prime_implicant_result prime_implicants_hellman(int num_bits, int num_trues, int *trues); 

#ifdef __cplusplus
}
#endif