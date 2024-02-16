#!/bin/bash

for i in LinearSystem-v0 Lorenz-v0 FluidFlow-v0 DoubleWell-v0
do for j in 60 80 100 120 140
        do for k in 100 200 300 400 500
                do for l in 83 103 123 143 163
                        do
                                flux submit ./random_interactions_job_script.sh $i $l $j $k
                        done
                done
        done
done
