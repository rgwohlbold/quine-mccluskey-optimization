#!/bin/sh
set -e

min_bits=1
max_bits=22
num_measurements=100
compilers="/usr/bin/gcc"
#compilers="/usr/bin/gcc /usr/bin/clang"
measurements_file="measurements_merge.csv"

make -j$(nproc)

[ -e "$measurements_file" ] && rm "$measurements_file"
echo "compiler_version,compiler_flags,cpu_model,implementation,bits,cycles" > "$measurements_file"

for compiler in $compilers; do
    # since we change the compiler, we have to set the other options in another cmake run
    cmake . -D CMAKE_C_COMPILER="$compiler"
    cmake . -D LOG_LEVEL=2
    make clean
    make -j $(nproc)
    implementations="$(./prime_implicants merge_implementations)"
    for i in $(seq $min_bits $max_bits); do
        for k in $(seq 1 "$num_measurements"); do
            for implementation in $implementations; do
                taskset --cpu-list 9 ./prime_implicants measure_merge "$implementation" "$i"
            done
        done
    done
done
python plot_merge.py
