#!/bin/sh
set -e

min_bits=1
max_bits=22
num_measurements=50
compilers="/usr/bin/gcc /usr/bin/clang"
measurements_file="measurements_merge.csv"
implementations="merge_implicants_bits merge_implicants_avx2 merge_implicants_pext"

make -j$(nproc)

[ -e "$measurements_file" ] && rm "$measurements_file"
echo "compiler_version,compiler_flags,cpu_model,implementation,bits,cycles,ops" > "$measurements_file"

for compiler in $compilers; do
    # since we change the compiler, we have to set the other options in another cmake run
    cmake . -D CMAKE_C_COMPILER="$compiler"
    cmake . -D LOG_LEVEL=2
    make clean
    make -j $(nproc)
    for k in $(seq 1 "$num_measurements"); do
        for i in $(seq $min_bits $max_bits); do
            for implementation in $implementations; do
                ./prime_implicants measure_merge "$implementation" "$i"
            done
        done
    done
done
python plot_merge.py
