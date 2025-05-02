#include "util.h"

#include <assert.h>
#include <stdlib.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif


// calculate 3**num_bits
int calculate_num_implicants(int num_bits) {
    int num_implicants = 1;
    for (int i = 0; i < num_bits; i++) {
        num_implicants *= 3;
    }
    return num_implicants;
}

bool *allocate_boolean_array(int num_elements) {
    bool *arr = (bool *)calloc(num_elements, sizeof(bool));
    if (arr == NULL) {
        perror("could not allocate boolean array");
        exit(EXIT_FAILURE);
    }
    return arr;
}

void flush_cache(bool *array, int num_elements) {
    const int block_size = 64;

    // align pointer to 64 bytes
    uint8_t *ptr = (uint8_t *) (((unsigned long)array) & ~(block_size - 1));
    uint8_t *end = (uint8_t *) &array[num_elements];
    #ifdef __x86_64__
    for (; ptr < end; ptr += block_size) {
        _mm_clflush(ptr);
    #endif
    #ifdef __aarch64__
        // For each 64‐byte line in [p, end):
    for (; ptr < end; ptr += block_size) {
        // Clean & Invalidate to Point of Coherency
        __asm__ volatile("dc civac, %0" :: "r"(ptr) : "memory");
    }
    // Ensure the DC instructions have completed
    __asm__ volatile("dsb ish" ::: "memory");
    // Synchronize the instruction stream (not strictly needed for data caches,
    // but often recommended after cache maintenance)
    __asm__ volatile("isb" ::: "memory");
    #endif
    
}


/* Generated using the following Python code:
for n in range(20):
    print("{", end='')
    for k in range(20):
        print(math.comb(n, k), end=',')
    print("},")
*/
const int binomial_coefficients[20][20] = {{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,4,6,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,5,10,10,5,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,6,15,20,15,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,7,21,35,35,21,7,1,0,0,0,0,0,0,0,0,0,0,0,0,},
{1,8,28,56,70,56,28,8,1,0,0,0,0,0,0,0,0,0,0,0,},
{1,9,36,84,126,126,84,36,9,1,0,0,0,0,0,0,0,0,0,0,},
{1,10,45,120,210,252,210,120,45,10,1,0,0,0,0,0,0,0,0,0,},
{1,11,55,165,330,462,462,330,165,55,11,1,0,0,0,0,0,0,0,0,},
{1,12,66,220,495,792,924,792,495,220,66,12,1,0,0,0,0,0,0,0,},
{1,13,78,286,715,1287,1716,1716,1287,715,286,78,13,1,0,0,0,0,0,0,},
{1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1,0,0,0,0,0,},
{1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1,0,0,0,0,},
{1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1,0,0,0,},
{1,17,136,680,2380,6188,12376,19448,24310,24310,19448,12376,6188,2380,680,136,17,1,0,0,},
{1,18,153,816,3060,8568,18564,31824,43758,48620,43758,31824,18564,8568,3060,816,153,18,1,0,},
{1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1,},
};

int binomial_coefficient(int n, int k) {
    assert(0 <= n);
    assert(0 <= k);
    assert(n < 20);
    assert(k < 20);
    return binomial_coefficients[n][k];
}