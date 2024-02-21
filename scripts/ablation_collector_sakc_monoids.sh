#!/bin/bash

# Input arguments
# 1: env-id

for i in 1 2 3 4
do for j in 1 2 3 4
  do
    
    echo ${i} ${j}

    python -m analysis.avg_performance_from_tensorboard \
          --path=/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/DataLogs/AblationAnalysis/SAKC/${1}/Monoids/StateOrder_${i}_ActionOrder_${j} \
          --mean-of-means=True
  done
done
