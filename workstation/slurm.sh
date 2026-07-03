#!/bin/bash

#SBATCH --partition long
#SBATCH --mem-per-cpu 8G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py --PD -f pd500_G1.txt -l1 0.25 -u1 4 -l2 0.125 -u2 2 -n 16 -G 1 -N 500
