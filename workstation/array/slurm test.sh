#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 64G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

source .venv/bin/activate
python abp_array.py --PD1 -f G1/test.txt -Ps 4 -Pf 7 -G 1 -T 1000000
