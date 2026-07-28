#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

source ../.venv/bin/activate
python collect.py -F "G 1" -tc timechain10000000.txt -Ps Ps_params.txt -Pf Pf_params.txt -o meanvx.txt --PD
