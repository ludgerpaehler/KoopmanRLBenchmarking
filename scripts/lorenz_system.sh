#!/bin/bash

# Bash script for the benchmarking of the linear system

# Generate the Koopman Tensor
python -m koopman_tensor.test_tensor --env-id=Lorenz-v0 --state-order=2 --action-order=2 --save-model

for i in 83 103 123 143 163
do

    # LQR
    python -m cleanrl.linear_quadratic_regulator \
	--env-id=Lorenz-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000

    # Discrete Value Iteration
    python -m cleanrl.discrete_value_iteration \
	--env-id=Lorenz-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000

    # SAC (Q)
    python -m cleanrl.sac_continuous_action \
	--env-id=Lorenz-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000

    # SAC (V)
    python -m cleanrl.value_based_sac_continuous_action \
	--env-id=Lorenz-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000

    # SKVI
    python -m cleanrl.value_based_sac_continuous_action \
	--env-id=Lorenz-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000 \
	--koopman

done
