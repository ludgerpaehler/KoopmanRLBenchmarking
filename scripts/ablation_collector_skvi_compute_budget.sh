#!/bin/bash

# Input arguments
# 1: env-id
# 2: folder-name

for i in 61 81 101 121 141
do for j in 90 110 130 150 170 190
  do
    
    echo ${i} ${j}

    python -m analysis.avg_performance_from_tensorboard \
          --path=/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/DataLogs/AblationAnalysis/SKVI/NumActionsNumTrainEpochs/${1}/NumActions_${i}_NumTrainEpochs_${j} \
          --mean-of-means=True
  done
done
