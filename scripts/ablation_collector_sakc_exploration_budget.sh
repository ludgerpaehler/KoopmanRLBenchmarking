#!/bin/bash

# Input arguments
# 1: env-id

for i in 60 80 100 120 140
do for j in 100 200 300 400 500
  do
    
    echo ${i} ${j}

    python -m analysis.avg_performance_from_tensorboard \
          --path=/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/DataLogs/AblationAnalysis/SAKC/${1}/ExplorationBudget/NumPaths_${i}_NumStepsPerPath_${j} \
          --mean-of-means=True
  done
done
