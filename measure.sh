#!/bin/sh

implementation="prime_implicants_dense"
min_bits=1
max_bits=18
num_measurements=1
measurements_file="measurements.csv"


[ -e "$measurements_file" ] && rm "$measurements_file"
echo "implementation,bits,cycles,ops" > "$measurements_file"
for k in $(seq 1 "$num_measurements"); do
    for i in $(seq $min_bits $max_bits); do
        ./prime_implicants measure "$implementation" "$i"
    done
done
python plot.py