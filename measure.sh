#!/bin/sh
set -e

min_bits=10
max_bits=20
num_measurements=10
compilers="/usr/bin/gcc"
#compilers="/usr/bin/gcc /usr/bin/clang"
measurements_file="measurements.csv"

[ -e "$measurements_file" ] && rm "$measurements_file"
echo "compiler_version,compiler_flags,cpu_model,implementation,bits,cycles,l1d_cache_misses,l1d_cache_accesses" > "$measurements_file"

for compiler in $compilers; do
    # since we change the compiler, we have to set the other options in another cmake run
    cmake . -D CMAKE_C_COMPILER="$compiler"
    cmake . -D LOG_LEVEL=2
    make clean
    make -j $(nproc) prime_implicants
    implementations="hellman bits bits_sp_load"
    for k in $(seq 1 "$num_measurements"); do
        for i in $(seq $min_bits $max_bits); do
            for implementation in $implementations; do
                if [ "$implementation" = "baseline" ]; then
                    continue
                fi
                ./prime_implicants measure "$implementation" "$i"
            done
        done
    done
done
python plot.py
