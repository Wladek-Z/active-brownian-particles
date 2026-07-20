#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 16G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python ../abp.py -Ps 1.5 -Pf 2.25 -N 2000 -G 1 --FPTD -T 200000
python ../abp.py -Ps 1.5 -Pf 2.625 -N 2000 -G 1 --FPTD -T 200000
python ../abp.py -Ps 1.5 -Pf 3 -N 2000 -G 1 --FPTD -T 200000
