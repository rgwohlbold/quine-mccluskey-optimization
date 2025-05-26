#!/bin/bash

testp="gen_tests"

# Ensure the directory exists
if [ ! -d "$testp" ]; then
    echo "Directory $testp does not exist. Please generate test files first."
    exit 1
fi

# Run ./prime_implicants test on all files in gen_tests
for testfile in "$testp"/*.txt; do
    if [ -f "$testfile" ]; then
        echo "Running ../prime_implicants test on $testfile"
        ../prime_implicants test "$testfile"
    else
        echo "No test files found in $testp."
        exit 1
    fi
done