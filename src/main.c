#include <stdlib.h>
#include <string.h>

#include "bitmap.h"
#include "debug.h"
#include "test.h"

void print_usage(char *argv[]) { LOG_INFO("usage: %s [test|measure|help|implementations|merge_implementations]", argv[0]); }
void print_test_usage(char *argv[]) {
    LOG_INFO("usage: %s test <testfile1> <testfile2> ...", argv[0]);
}
void print_test_single_usage(char *argv[]) {
    LOG_INFO("usage: %s test_single <implementation> <testfile1> <testfile2> ...", argv[0]);
}
void print_measure_usage(char *argv[]) { LOG_INFO("usage: %s measure <implementation> <num_bits>", argv[0]); }

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
    if (argc <= 1) {
        print_usage(argv);
    } else if (strcmp(argv[1], "test") == 0) {
        bitmap_test();
        if (argc <= 2) {
            print_test_usage(argv);
            return 0;
        }
        test_implementations(&argv[2], argc - 2);  // Pass the test files to the test_implementations function
    } else if (strcmp(argv[1], "test_single") == 0) {
        //bitmap_test();
        if (argc <= 3) {
            print_test_single_usage(argv);
            return 0;
        }
        test_implementation_single(argv[2], &argv[3], argc - 3);  // Pass the test files to the test_implementation function
    } else if (strcmp(argv[1], "implementations") == 0) {
        print_implementations();
    } else if (strcmp(argv[1], "merge_implementations") == 0) {
        print_merge_implementations();
    } else if (strcmp(argv[1], "measure") == 0) {
        if (argc <= 3) {
            print_measure_usage(argv);
        } else {
            const char *implementation = argv[2];
            int num_bits = parse_int(argv[3]);
            measure_implementations(implementation, num_bits);
        }
    } else if (strcmp(argv[1], "measure_merge") == 0) {
        if (argc <= 3) {
            print_measure_usage(argv);
        } else {
            const char *implementation = argv[2];
            int num_bits = parse_int(argv[3]);
            measure_merge(implementation, num_bits);
        }

    } else {
        print_usage(argv);
    }
}
