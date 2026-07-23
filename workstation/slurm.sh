#!/bin/bash

#SBATCH --partition long
#SBATCH --mem-per-cpu 64G
#SBATCH --time 48:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py --PD -f pdbig_G1.txt -l1 0.25 -u1 8 -l2 0.125 -u2 4 -n 32 -G 1 -N 1000
