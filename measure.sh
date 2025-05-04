#!/bin/sh

make

implementations="$(./prime_implicants implementations)"
min_bits=1
max_bits=18
num_measurements=1
measurements_file="measurements.csv"


[ -e "$measurements_file" ] && rm "$measurements_file"
echo "implementation,bits,cycles,ops" > "$measurements_file"
for implementation in $implementations; do
    for k in $(seq 1 "$num_measurements"); do
        for i in $(seq $min_bits $max_bits); do
            ./prime_implicants measure "$implementation" "$i"
        done
    done
done
python plot.py