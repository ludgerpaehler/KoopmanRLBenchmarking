#!/bin/bash

# Bash script to generate the Koopman tensors for bases of the order 1-4
# for which the order of the states, as well as the actions is varied

for i in LinearSystem-v0 Lorenz-v0 FluidFlow-v0 DoubleWell-v0
do for j in 60 80 100 120 140
    do for k in 100 200 300 400 500
        do

            python -m koopman_tensor.test_tensor \
            --env-id=$i \
            --num-paths=$j \
            --num-steps-per-path=$k \
            --save-model \
            --tensor-name=num_paths_${j}_num_steps_per_path_${k}

        done 
    done 
done

# For the plotting
#   x-axis: number of paths
#   y-axis: number of steps per path
#   
#   To be plotted as a heat-map
#       - the heat-value will be given by the mean episodic return
