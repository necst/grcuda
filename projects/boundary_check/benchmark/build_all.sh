#!/bin/bash

echo "Build the CUDA kernel examples using different optimization level"

opt_levels=( O0 O2 )
simplify=( y '' )
execution_mode=( 1 )
print_oob=( 1 )

for o in "${opt_levels[@]}"; do
    for s in "${simplify[@]}"; do
        make -s OPT_LEVEL=$o SIMPLIFY=$s EXECUTION_MODE=$execution_mode PRINT_OOB_ACCESSES=$print_oob
        make unmodified_kernels OPT_LEVEL=$o SIMPLIFY=$s
        make unsafe_kernels OPT_LEVEL=$o SIMPLIFY=$s
    done
done
