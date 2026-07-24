#!/bin/bash

#SBATCH --partition long
#SBATCH --mem-per-cpu 256G
#SBATCH --time 72:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py --PDA -f pd5M_G0.5.txt -l1 0.1875 -u1 6 -l2 0.125 -u2 4 -n 32 -G 0.5 -N 1000 -T 5000000
