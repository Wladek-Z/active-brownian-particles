#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 8G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py --hist -F 'hist_2.5_0.5_G1' -Ps 25 -Pf 12.5 -G 1 -N 1000
