#!/bin/bash
set -e

# List of all implementations to measure
implementations=(
#baseline
# bits
# pext
# pext_sp
pext_sp_unroll
pext_sp_unroll_ilp
# pext_sp_block
# pext_sp_block2
# pext_sp_block4
# pext_sp_block8
# pext_sp_load_block
# pext_sp_load_block2
# pext_sp_load_block4
# pext_sp_load_block8
# hellman
# avx2_sp_load_block2
# avx2_sp_load_block4
# avx2_sp_load_block8
# avx2_sp_load_block16
# avx512_sp_load_block2
# avx512_sp_load_block4
# avx512_sp_load_block8
# avx512_sp_load_block16
)

# Clean up any previously generated plots and data
echo "Running measurements for all implementations..."

# Forward all implementations as command line arguments
./measure_individual_impl.sh "${implementations[@]}"
