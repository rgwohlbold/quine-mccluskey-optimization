#!/bin/sh
set -e

# === USAGE CHECK ===
if [ $# -ne 1 ]; then
    echo "Usage: $0 <implementation>"
    exit 1
fi

implementation="$1"

min_bits=1
max_bits=22
num_measurements=5
compilers="/usr/bin/clang"
measurements_file="measurements.csv"


[ -e "$measurements_file" ] && rm "$measurements_file"
[ -e "m_${implementation}.csv" ] && rm "m_${implementation}.csv"
echo "compiler_version,compiler_flags,cpu_model,implementation,bits,cycles,ops" > "$measurements_file"

for compiler in $compilers; do
    # since we change the compiler, we have to set the other options in another cmake run
    cmake . -D CMAKE_C_COMPILER="$compiler"
    cmake . -D LOG_LEVEL=2
    make clean
    make -j $(nproc)

    for k in $(seq 1 "$num_measurements"); do
        for i in $(seq $min_bits $max_bits); do
            taskset --cpu-list 9 ./prime_implicants measure "$implementation" "$i"
        done
        echo " === Measurement $k done ===\n"
    done

    cp "measurements.csv" "m_${implementation}.csv"

done
# python3 plot.py
