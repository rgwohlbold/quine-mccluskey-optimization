#!/bin/bash
set -e

# List of all implementations to measure
implementations=(
# baseline
bits
pext
pext_sp
pext_sp_intra
# pext_sp_intra_ilp
pext_sp_inter2
# pext_sp_inter4
# pext_sp_inter8
pext_sp_load_inter2
# pext_sp_load_inter4
# pext_sp_load_inter8
# hellman
# avx2_sp_load_inter2
# avx2_sp_load_inter4
# avx2_sp_load_inter8
# avx512_sp_load_inter2
# avx512_sp_load_inter4
# avx512_sp_load_inter8
)

# Clean up any previously generated plots and data
echo "Running measurements for all implementations..."

# Forward all implementations as command line arguments
./measure_individual_impl.sh "${implementations[@]}"
