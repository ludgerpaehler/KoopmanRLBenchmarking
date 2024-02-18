#!/bin/bash

for i in FluidFlow-v0 Lorenz-v0 DoubleWell-v0
do for j in 83 103 123 143 163
  do
      python -m evaluations.value_based_sac_continuous_action_eval \
        --env-id=$i \
        --exp-name=koopman_${j}_eval \
        --total-timesteps=200000 \
        --state-dict=/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/DataLogs/Interpretability_Table/saved_models/${i}/${j}/value_based_sakc_actor.pt \
        --koopman
  done
done
