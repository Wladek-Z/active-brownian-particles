#!/bin/bash

for Ps in $(seq 0.25 0.25 4);
do
    for Pf in $(seq 0.5 0.5 8);
    do
        echo $Ps >> Ps_params2.txt
        echo $Pf >> Pf_params2.txt
    done
done
