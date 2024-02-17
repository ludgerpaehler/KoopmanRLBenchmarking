#!/bin/bash

# Flux parameters
#flux: -N 1
#flux: --exclusive
#flux: -t 45

# Input to the bash script
# 1: env-id
# 2: seed
# 3: extracted value function

# Source the virtual environment
#source koopman_bench_env/bin/activate

# Move into the KoopmanRLBenchmarking folder
#cd KoopmanRLBenchmarking

# Run the actual RL experiment
python -m cleanrl.interpretability_discrete_value_iteration \
--end-id="$1" \
--alpha=1 \
--seed=$2 \
--total-timesteps=50000 \
--value-fn-weights=$3
