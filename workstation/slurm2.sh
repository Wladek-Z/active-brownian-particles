#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 16G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py -Ps 1.5 --eff -f eff_n32_1.5_1.npz -N 1000 -G 1 -l1 0.125 -u1 2 -n 32
