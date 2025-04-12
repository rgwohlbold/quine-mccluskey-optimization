#include <stdlib.h>
#include <string.h>
#include "debug.h"
#include "test.h"

void print_usage(char *argv[]) {
    LOG_INFO("usage: %s [test|measure|help]", argv[0]);
}

int main(int argc, char *argv[]) { 
    if (argc <= 1 || strcmp(argv[1], "test") == 0) {
        test_implementations();
    } else if (strcmp(argv[1], "measure") == 0) {
        measure_implementations();
    } else {
        print_usage(argv);
    }
}