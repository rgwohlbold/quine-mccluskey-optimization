#include <stdlib.h>
#include <string.h>
#include "debug.h"
#include "test.h"

void print_usage(char *argv[]) {
    LOG_INFO("usage: %s [test|measure|help]", argv[0]);
}

void print_measure_usage(char *argv[]) {
    LOG_INFO("usage: %s measure <implementation> <num_bits>", argv[0]);
}

int parse_int(const char *s) {
    int x = 0;
    int res = sscanf(s, "%d", &x);
    if (!res) {
        LOG_INFO("could not parse integer \"%s\"", s);
        exit(EXIT_FAILURE);
    }
    return x;
}

int main(int argc, char *argv[]) { 
    if (argc <= 1 || strcmp(argv[1], "test") == 0) {
        test_implementations();
    } else if (strcmp(argv[1], "measure") == 0) {
        if (argc <= 3) {
            print_measure_usage(argv);
        } else {
            const char *implementation = argv[2];
            int num_bits = parse_int(argv[3]);
            measure_implementations(implementation, num_bits);
        }
    } else if (strcmp(argv[1], "measure_merge") == 0) {
        if (argc <= 2) {
            print_measure_usage(argv);
        } else {
            int num_bits = parse_int(argv[2]);
            measure_merge(num_bits);
        }
        
    } else {
        print_usage(argv);
    }
}