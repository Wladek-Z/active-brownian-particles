#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 16G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py --PD -f pd_G0.5.txt -l1 0.25 -u1 4 -l2 0.125 -u2 2 -n 16 -G 0.5 -N 1000
