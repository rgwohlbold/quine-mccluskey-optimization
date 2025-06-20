#include <assert.h>
#include <string.h>
#include "bitmap.h"
#include "debug.h"
#include "util.h"

bitmap bitmap_allocate(size_t num_bits) {
    size_t num_bytes = (num_bits + 7) / 8 + 64; // add 64 bytes to make sure we are aligned
    uint8_t *bits = (uint8_t *) calloc(num_bytes, sizeof(uint8_t));
    if (bits == NULL) {
        perror("cannot allocate bitmap");
        exit(EXIT_FAILURE);
    }
    bitmap result = {
        .num_bits = num_bits,
        .malloc_ptr = bits,
        .bits = bits + (64 - ((uintptr_t)bits & 0x3F)), // align to 64 bytes
    };
    // access all pages of the bitmap once to make the OS back them with physical memory
    for (size_t i = 0; i < num_bits; i += 4096 * 8) {
        BITMAP_SET_FALSE(result, i);
    }
    return result;
}

void bitmap_free(bitmap map) {
    free(map.malloc_ptr);
}

bool bitmap_cmp(bitmap map1, bitmap map2) {
    if (map1.num_bits != map2.num_bits) {
        return false;
    }
    for (size_t i = 0; i < map1.num_bits - map1.num_bits % 64; i += 64) {
        if (((uint64_t *)(map1.bits))[i/64] != ((uint64_t *)(map2.bits))[i/64]) {
            return false;
        }
    }
    for (size_t i = map1.num_bits - map1.num_bits % 64; i < map1.num_bits; i++) {
        if (BITMAP_CHECK(map1, i) ^ BITMAP_CHECK(map2, i)) {
            return false;
        }
    }
    return true;
}

uint64_t bitmap_implicant_to_index(int num_bits, const char *s) {
    // count number of dashes to get section index
    uint64_t num_dashes = 0;
    uint64_t section_offset = 0;
    for (int i = 0; i < num_bits; i++) {
        if (s[i] == '-') {
            section_offset += binomial_coefficient(num_bits, num_dashes) * (1 << (num_bits - num_dashes));
            num_dashes++;
        }
    }

    // look at dash pattern to get chunk offset
    uint64_t chunk_offset = 0;
    uint64_t dashes_remaining = num_dashes;
    for (int i = num_bits-1; i >= 0; i--) {
        if (s[i] == '-') {
            dashes_remaining -= 1;
        } else if (dashes_remaining >= 1) {
            chunk_offset += binomial_coefficient(i, dashes_remaining-1);
        }
    }
    chunk_offset *= 1 << (num_bits - num_dashes);

    // look at pattern of 0s and 1s to get offset within chunk
    uint64_t offset_within_chunk = 0;
    for (uint64_t i = 0; i < num_bits; i++) {
        if (s[i] == '0') {
            offset_within_chunk *= 2;
        } else if (s[i] == '1') {
            offset_within_chunk *= 2;
            offset_within_chunk += 1;
        }

    }
    //LOG_DEBUG("implicant %s section_index=%d chunk_offset=%d offset_within_chunk=%d", s, section_offset, chunk_offset, offset_within_chunk);
    return section_offset + chunk_offset + offset_within_chunk;
}

void bitmap_index_to_implicant(int num_bits, size_t bitset_index, char *s) {
    size_t num_dashes = 0;
    size_t section_offset = 0;
    for (int64_t remaining_bits = num_bits; remaining_bits >= 0; remaining_bits--) {
        size_t new_section_offset = section_offset + (1 << remaining_bits) * binomial_coefficient(num_bits, num_dashes);
        if (new_section_offset > bitset_index) {
            break;
        }
        section_offset = new_section_offset;
        num_dashes++;
    }
    size_t remaining_bits = num_bits - num_dashes;
    size_t chunk_index = (bitset_index - section_offset) / (1 << remaining_bits);
    size_t offset_within_chunk = (bitset_index - section_offset) % (1 << remaining_bits);
    //LOG_DEBUG("bitset_index %d section_offset=%d chunk_index=%d offset_within_chunk=%d", bitset_index, section_offset, chunk_index, offset_within_chunk);

    size_t dashes_set = 0;
    for (int i = 0; i < num_bits; i++) {
        size_t dashes_remaining = num_dashes - dashes_set;
        if (dashes_remaining == 0) {
            s[num_bits-i-1] = '.';
        } else {
            size_t possibilities_if_dash_is_set = binomial_coefficient(num_bits-i-1, dashes_remaining-1);
            if (chunk_index < possibilities_if_dash_is_set) {
                s[num_bits-i-1] = '-';
                dashes_set++;
            } else {
                s[num_bits-i-1] = '.';
                chunk_index -= possibilities_if_dash_is_set;
            }
        }
    }
    // put in value of implicant except for dashes
    size_t bit = 0;
    for (int k = 0; k < num_bits; k++) {
        size_t bitmask = 1 << bit;
        if (s[num_bits - k - 1] == '.') {
            if ((offset_within_chunk & bitmask) != 0) {
                s[num_bits - k - 1] = '1';
            } else {
                s[num_bits - k - 1] = '0';
            }
            bit++;
        }
    }
}

static void test_bitmap_n(size_t n) {
    LOG_INFO("testing bitmap n=%d", n);
    bitmap map = bitmap_allocate(n);

    // test number of bits
    assert(map.num_bits == n);
    for (size_t i = 0; i < n; i++) {
        assert(!BITMAP_CHECK(map, i));
    }

    // set ith bit
    for (size_t i = 0; i < n; i++) {
        BITMAP_SET_TRUE(map, i);
        assert(BITMAP_CHECK(map, i));
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                assert(!BITMAP_CHECK(map, k));
            }
        }
        BITMAP_SET_FALSE(map, i);
        assert(!BITMAP_CHECK(map, i));
    }

    bitmap_free(map);
}

static void test_cmp_bitmap() {
    LOG_INFO("testing bitmap comparison");
    bitmap map1 = bitmap_allocate(32);
    bitmap map2 = bitmap_allocate(32);
    assert(bitmap_cmp(map1, map2));
    BITMAP_SET_TRUE(map1, 3);
    assert(!bitmap_cmp(map1, map2));
    BITMAP_SET_TRUE(map2, 5);
    assert(!bitmap_cmp(map1, map2));
    BITMAP_SET_TRUE(map1, 5);
    BITMAP_SET_TRUE(map2, 3);
    assert(bitmap_cmp(map1, map2));
    bitmap_free(map1);
    bitmap_free(map2);
}

static void test_implicant_bitmap_index() {
    struct {
        int num_bits;
        const char *implicant;
        uint64_t index;
    } tests[] = {
        {4, "0000", 0},
        {4, "0001", 1},
        {4, "1110", 14},
        {4, "000-", 16},
        {4, "101-", 21},
        {4, "1-11", 39},
        {4, "11--", 51},
        {4, "-1-0", 66},
        {4, "-1--", 75},
        {4, "----", 80},
    };
    LOG_INFO("testing implicant bitmap mapping")
    char implicant[5]; // adjust size when adding test cases with more bits
    implicant[sizeof(implicant)-1] = '\0';
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        assert(bitmap_implicant_to_index(tests[i].num_bits, tests[i].implicant) == tests[i].index);
    }
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        bitmap_index_to_implicant(tests[i].num_bits, tests[i].index, implicant);
        assert(strncmp(implicant, tests[i].implicant, tests[i].num_bits) == 0);
    }
}

static void test_implicant_bitmap_index_inversion(int num_bits) {
    LOG_INFO("testing implicant bitmap mapping inversion num_bits=%d", num_bits);
    char implicant[num_bits+1];
    implicant[num_bits] = '\0';
    int num_implicants = calculate_num_implicants(num_bits);
    for (int i = 0; i < num_implicants; i++) {
        bitmap_index_to_implicant(num_bits, i, implicant);
        int index = bitmap_implicant_to_index(num_bits, implicant);
        assert(i == index);
    }
}

void bitmap_test() {
    int sizes[] = {1, 5, 7, 8, 13, 16, 27, 32, 35};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        test_bitmap_n(sizes[i]);
    }
    test_cmp_bitmap();
    test_implicant_bitmap_index();
    test_implicant_bitmap_index_inversion(10);
}
