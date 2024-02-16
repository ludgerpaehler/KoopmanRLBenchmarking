#!/bin/bash

for i in LinearSystem-v0 Lorenz-v0 FluidFlow-v0 DoubleWell-v0
do for j in 1 2 3 4
	do for k in 1 2 3 4
                do for l in 83 103 123 143 163
                        do
                                flux submit ./monoids_job_script.sh $i $l $j $k
                        done
                done
        done
done
