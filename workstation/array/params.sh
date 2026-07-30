#!/bin/bash

for Ps in $(seq 0.25 0.25 4);
do
    for Pf in $(seq 0.5 0.5 8);
    do
        if awk -v ps="$Ps" -v pf="$Pf" 'BEGIN { exit !(pf == 2 * ps - 1) }'; then
            echo "$Ps" >> Ps_diffusive.txt
            echo "$Pf" >> Pf_diffusive.txt
        fi
    done
done
