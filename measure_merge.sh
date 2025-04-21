#!/bin/sh

min_bits=1
max_bits=19
num_measurements=50
measurements_file="measurements_merge.csv"

make -j$(nproc)

[ -e "$measurements_file" ] && rm "$measurements_file"
echo "implementation,bits,cycles,ops" > "$measurements_file"
for k in $(seq 1 "$num_measurements"); do
    for i in $(seq $min_bits $max_bits); do
        ./prime_implicants measure_merge "$i"
    done
done
python plot_merge.py