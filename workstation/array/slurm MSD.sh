#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-256
#
#######################################

G=1
offset=0
output="G ${G} results/MSD o${offset}"

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

mkdir -p "${output}"

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python 'get stats.py' --MSD -F "G ${G}/${x} ${y}" -Ps $x -Pf $y -o "${output}" -f "lags o${offset}.npz" -off ${offset}
