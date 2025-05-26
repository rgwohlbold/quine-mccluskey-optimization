#include "debug.h"

#include <stdarg.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "implicant.h"
#include "bitmap.h"

int _log_fmt(const char *prefix, const char *file, int line, const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("%s %s:%d: ", prefix, file, line);
    vprintf(format, args);
    printf("\n");
    va_end(args);
    return 0;
}

void print_bitmap(const char *msg, const bitmap *map) {
    char *buffer = malloc(map->num_bits + 1); // Allocate memory for the bitmap string (+1 for null terminator)
    if (!buffer) {
        LOG_DEBUG("Failed to allocate memory for bitmap printing.");
        return;
    }
    for (size_t i = 0; i < map->num_bits; i++) {
        buffer[i] = BITMAP_CHECK((*map), i) ? '1' : '0'; // Convert each bit to '1' or '0'
    }
    buffer[map->num_bits] = '\0'; // Null-terminate the string

    LOG_DEBUG("%s: %s", msg, buffer); // Print the message and the bitmap in one line

    free(buffer); // Free the allocated memory
}



// Print bitmap in sparse format.
// Print indices of ones in bitmap, separated by spaces
// Use LOG_DEBUG just once
void print_bitmap_sparse(const char *msg, const bitmap *map) {
    LOG_DEBUG("%s: ", msg);
    for (size_t i = 0; i < map->num_bits; i++) {
        if (BITMAP_CHECK((*map), i)) {
            printf("%lu ", i);
        }
    }
    printf("\n");
}

void print_primes_sparse(const char *msg, const bitmap* implicants, const bitmap* merged){
    LOG_DEBUG("%s: ", msg);
    for (size_t i = 0; i < implicants->num_bits; i++) {
        if (BITMAP_CHECK((*implicants), i) && !BITMAP_CHECK((*merged), i)) {
            printf("%lu ", i);
        }
    }
    printf("\n");
}


#ifdef __AVX2__
void log_m256i(const char *msg, const __m256i *value) {
    uint64_t buffer[4];
    _mm256_storeu_si256((__m256i*)buffer, *value);
    LOG_DEBUG("%s %lx %lx %lx %lx", msg, buffer[3], buffer[2], buffer[1], buffer[0]);
}

#endif

#ifdef __aarch64__
void log_u64x2(const char *msg, const uint64x2_t *value) {
    uint64_t buffer[2];
    vst1q_u64(buffer, *value);
    LOG_DEBUG("%s %lx %lx", msg, buffer[1], buffer[0]);
}
void log_u64x2_binary(const char *msg, const uint64x2_t *value) {
    uint64_t buf[2];
    vst1q_u64(buf, *value);

    // buf[1] = high 64 bits, buf[0] = low 64 bits
    char bits[128 + 1];
    for (int i = 0; i < 64; i++) {
        // extract bit (63-i) of the high half
        bits[i] = (buf[1] & (1ULL << (63 - i))) ? '1' : '0';
    }
    for (int i = 0; i < 64; i++) {
        // extract bit (63-i) of the low half
        bits[64 + i] = (buf[0] & (1ULL << (63 - i))) ? '1' : '0';
    }
    bits[128] = '\0';

    // now print it in one go
    LOG_DEBUG("%s %s", msg, bits);
}

// Function to print the first n bits of a bitmap
void print_bitmap_bits(const bitmap *bmp, size_t n) {
    size_t num_bits = n > bmp->num_bits ? bmp->num_bits : n; // Ensure n does not exceed bitmap size
    for (size_t i = 0; i < num_bits; i++) {
        // Check if the bit at position i is set
        int bit = BITMAP_CHECK((*bmp), i) ? 1 : 0;
        printf("%d", bit);

        // Add a space every 8 bits for readability
        if ((i + 1) % 8 == 0) {
            printf(" ");
        }
    }
    printf("\n");
}
#endif
