#!/bin/bash

G=$1

#SBATCH --partition short
#SBATCH --mem-per-cpu 1G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-400
#
#######################################

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

mkdir -p "G ${G}/${x} ${y}"

source ../.venv/bin/activate
python abp_log.py -F "G ${G}/${x} ${y}" -tc timechain10000000.txt -Ps $x -Pf $y -G $G -N 1000
