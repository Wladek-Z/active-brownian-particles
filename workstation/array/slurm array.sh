#!/bin/bash

G=$1

#SBATCH --partition medium
#SBATCH --mem-per-cpu 64G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-400
#
#######################################

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

source .venv/bin/activate
python abp_log.py -F 'G ${G}' -Ps $x -Pf $y -G $G -N 1000
