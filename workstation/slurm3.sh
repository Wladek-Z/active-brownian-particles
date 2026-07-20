#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 128G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python ../abp.py -Ps 1.5 -Pf 1.875 -N 2000 -G 1 --FPTD -T 2000000
