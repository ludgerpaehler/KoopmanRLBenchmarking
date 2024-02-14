#!/bin/bash

# Bash script for the benchmarking of the linear system

# Generate the Koopman Tensor
python -m koopman_tensor.test_tensor --env-id=LinearSystem-v0 --state-order=2 --action-order=2 --save-model

for i in 83 103 123 143 163
do

    # SKVI
    python -m cleanrl.value_based_sac_continuous_action \
	--env-id=LinearSystem-v0 \
	--alpha=1 \
	--seed=$i \
	--total-timesteps=50000 \
	--koopman

done
