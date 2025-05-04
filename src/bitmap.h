#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
    int num_bits;
    uint8_t *bits;
} bitmap;

bitmap bitmap_allocate(int num_bits);
void bitmap_free(bitmap map);
bool bitmap_cmp(bitmap map1, bitmap map2);
void bitmap_test();
int bitmap_implicant_to_index(int num_bits, const char *s);
void bitmap_index_to_implicant(int num_bits, int bitset_index, char *s);

#define BITMAP_CHECK(map, index) (((map.bits)[(index)/8] >> ((index) % 8)) & 1)
#define BITMAP_SET_TRUE(map, index) ((map.bits)[(index)/8] |= 1 << ((index) % 8))
#define BITMAP_SET_FALSE(map, index) ((map.bits)[(index)/8] &= ~(1 << ((index) % 8)))
#define BITMAP_SET(map, index, value) ((value) ? BITMAP_SET_TRUE((map), (index)) : BITMAP_SET_FALSE((map), (index)))


