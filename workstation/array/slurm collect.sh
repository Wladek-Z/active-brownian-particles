#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-256
#
#######################################

G=1

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

output="G ${G} NV results/mean velocities"
mkdir -p "${output}"

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python collect.py -i "/data/biophys/ABP_channel/G ${G} NV/${x} ${y}" -Ps $x --VX -o "${output}/${x} ${y}.txt"
