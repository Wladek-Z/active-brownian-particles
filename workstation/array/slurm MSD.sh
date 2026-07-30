#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-14
#
#######################################

G=1
o=0

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_diffusive.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_diffusive.txt)

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python '.\get stats.py' --MSD -F "G ${G}/${x} ${y}" -Ps $x -Pf $y -o "G ${G} results/MSD" -f lags.npz 
