#!/bin/bash

# Bash script to generate the Koopman tensors for bases of the order 1-4
# for which the order of the states, as well as the actions is varied

for i in 1 2 3 4
do for j in LinearSystem-v0 Lorenz-v0 FluidFlow-v0 DoubleWell-v0
    do for k in 1 2 3 4
        do

            python -m koopman_tensor.test_tensor \
            --env-id=$j \
            --state-order=$i \
            --action-order=$k \
            --save-model \
            --tensor-name=state_order_${i}_action_order_${k}

        done 
    done 
done

# For the plotting
#   x-axis: state-order
#   y-axis: action-order
#   
#   To be plotted as a heat-map
#       - the heat-value will be given by the mean episodic return
