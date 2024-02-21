#!/bin/bash

# Input arguments
# 1: env-id
# 2: folder-name

for i in LinearSystem FluidFlow Lorenz DoubleWell
do for j in 8192 12288 16384 20480 24576
  do
    
    echo ${i} ${j}

    python -m analysis.avg_performance_from_tensorboard \
          --path=/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/DataLogs/AblationAnalysis/SKVI/Batch_Size/${i}/${j} \
          --mean-of-means=True
  done
done
