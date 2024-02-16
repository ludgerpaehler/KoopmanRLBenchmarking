#!/bin/bash

# Flux arguments
#flux: -N 1
#flux: --exclusive
#flux: -t 240

# Bash script to generate the Koopman tensor for bases of where the
# number of paths, as well as the number of steps per path is varied

# Args:
# 1: Environment name

# Source the virtual environment
source koopman_bench_env/bin/activate

# Move into the KoopmanRLBenchmarking folder
cd KoopmanRLBenchmarking

# Run the actual Koopman tensors
for i in 50 60 70 80 90 100 110 120 130 140 150
do for j in 50 100 150 200 250 300 350 400 450 500 550
        do

                python -m koopman_tensor.test_tensor \
                --env-id="$1" \
                --num-paths=$i \
                --num-steps-per-path=$j \
                --save-model \
                --tensor-name=num_path_${i}_num_steps_per_path_${j}
        done
done
