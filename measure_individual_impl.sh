#!/bin/sh
set -e

# === USAGE CHECK ===
if [ $# -lt 1 ]; then
    echo "Usage: $0 <implementation1> [<implementation2> ...]"
    exit 1
fi

implementations="$@"

min_bits=1
max_bits=22
num_measurements=10
compilers="/usr/bin/clang /usr/bin/gcc"
measurements_file="measurements.csv"


for compiler in $compilers; do
    # since we change the compiler, we have to set the other options in another cmake run
    cmake . -D CMAKE_C_COMPILER="$compiler"
    cmake . -D LOG_LEVEL=2 -DGENERATE_ASM=OFF
    make clean
    make -j $(nproc)

    # compiler_suffix is the compiler name without the path
    compiler_suffix=$(basename "$compiler")

    # for each implementation, we run the measurements
    for implementation in $implementations; do

        [ -e "$measurements_file" ] && rm "$measurements_file"
        [ -e "m_${implementation}_${compiler_suffix}.csv" ] && rm "m_${implementation}_${compiler_suffix}.csv"
        echo "compiler_version,compiler_flags,cpu_model,implementation,bits,cycles,l1d_cache_misses,l1d_cache_accesses" > "$measurements_file"

        for k in $(seq 1 "$num_measurements"); do
            for i in $(seq $min_bits $max_bits); do
                taskset --cpu-list 9 ./prime_implicants measure "$implementation" "$i"
            done
            echo " === Measurement $k done ===\n"
        done

        if [ "$implementation" = "hellman" ]; then
            # For hellman, remove the whole generator expression (anything like <$...>)
            sed 's/\$<\$<COMPILE_LANGUAGE:C>:-fno-tree-vectorize>//g; s/\$<\$<COMPILE_LANGUAGE:C>:-fno-vectorize>//g' \
                "$measurements_file" > "m_${implementation}_${compiler_suffix}.csv"
        else
            # For others, replace with the clean flag
            sed 's/\$<\$<COMPILE_LANGUAGE:C>:-fno-tree-vectorize>/-fno-tree-vectorize/g; s/\$<\$<COMPILE_LANGUAGE:C>:-fno-vectorize>/-fno-vectorize/g' \
                "$measurements_file" > "m_${implementation}_${compiler_suffix}.csv"
        fi
        # cp "measurements.csv" "m_${implementation}_${compiler_suffix}.csv"
    done
done
