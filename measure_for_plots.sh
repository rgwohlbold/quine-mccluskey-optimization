#!/bin/bash
# filepath: /home/akili/ETH/Advanced Systems Lab/Project/team58/measure_for_plots.sh
set -e

# List of all implementations to measure
implementations=(
  #"baseline"
  # "bits"
  # "bits_sp"
  # "hellman"
  # "pext"
  # "avx2"
  # "avx2_sp"
  # "avx2_sp_ssa"
  "avx2_sp_ilp"
  "avx2_sp_unroll"
  "avx2_sp_shuffle"
  "avx2_sp_load_shuffle"
  "avx512_sp_block"
  "avx512_sp_load_block"
)

# Clean up any previously generated plots and data
echo "Running measurements for all implementations..."

# Run measurements for each implementation
for impl in "${implementations[@]}"; do
  echo "===== Measuring implementation: $impl ====="
  ./measure_individual_impl.sh "$impl"
  echo "===== Completed measurements for: $impl ====="
  echo
done

# Construct the arguments for the plotting script
plot_args=""
for impl in "${implementations[@]}"; do
  plot_args+="m_${impl}.csv "
done

# Run the plotting script with all the generated CSV files
echo "===== Generating plots ====="
python3 plot_multiple.py $plot_args

echo "===== All done! ====="