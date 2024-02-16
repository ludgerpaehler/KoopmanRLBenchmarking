#!/bin/bash

for i in LinearSystem-v0 Lorenz-v0 FluidFlow-v0 DoubleWell-v0
do
        flux submit ./space_of_random_interactions.sh $i
done
