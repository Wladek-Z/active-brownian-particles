#!/bin/bash

G="$1"
mkdir G${G}_data

for Ps in $(seq 0.2 0.2 4);
do
    for Pf in $(seq 0.4 0.4 8);
    do
        echo $Ps >> Ps_params.txt
        echo $Pf >> Pf_params.txt
    done
done