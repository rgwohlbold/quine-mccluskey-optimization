#!/bin/bash

METRICS=(
    "cycles"
    "L1-dcache-load-misses"
    "L1-dcache-loads"
    "LLC-load-misses"
    "LLC-store-misses"
)

IMPLEMENTATIONS=(
    "native_dfs_sp"
    "native_dfsb_sp"
)
NUM_BITS=(
    16
    17
    18
    19
    20
    21
)
BASE_CMD="sudo perf stat --cpu 9 -e 'cycles,L1-dcache-load-misses,LLC-load-misses,LLC-store-misses' -- ../prime_implicants measure"

NUM_ITER=12
outfile="perf_results_$(date +%s).csv"
# Measure all of them 10 times, write into csv
echo -n "cmd,repetition,nbits" >$outfile
for metric in "${METRICS[@]}"; do
    echo -n ",$metric" >>$outfile
done
echo "" >>$outfile
for nbits in "${NUM_BITS[@]}"; do
    for j in "${!IMPLEMENTATIONS[@]}"; do
        for i in $(seq 1 $NUM_ITER); do
            cmd="${BASE_CMD} ${IMPLEMENTATIONS[$j]} $nbits"
            echo "Running: $cmd"
            output=$(eval $cmd 2>&1)
            if [ $? -ne 0 ]; then
                echo "Command failed: $cmd"
                continue
            fi
            echo -n "${IMPLEMENTATIONS[$j]},$i,$nbits" >>$outfile
            for metric in "${METRICS[@]}"; do
                if ! echo "$output" | grep -q "/$metric/"; then
                    echo "Metric $metric not found in output for ${IMPLEMENTATIONS[$j]} on iteration $i"
                    continue
                fi
                m_num=$(echo "$output" | grep "/$metric/" | awk '{print $1}')
                echo -n ",$m_num" >>$outfile
            done
            echo "" >>$outfile
        done
    done
done
