#!/bin/bash

# Flux parameters
#flux: -N 1
#flux: --exclusive
#flux: -t 45

# Inputs to the bash script:
#  1: env-id
#  2: seed
#  3: state-order
#  4: action-order

# Source the virtual environment
source koopman_bench_env/bin/activate

# Move into the KoopmanRLBenchmarking folder
cd KoopmanRLBenchmarking

# Run the actual RL training
python -m cleanrl.value_based_sac_continuous_action \
--env-id="$1" \
--alpha=1 \
--seed=$2 \
--total-timesteps=50000 \
--koopman-tensor=state_order_${3}_action_order_${4} \
--koopman-name-arg=state_order_${3}_action_order_${4} \
--koopman
