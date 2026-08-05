#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-14
#
#######################################

G=1

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_diffusive.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_diffusive.txt)

output="G ${G} results/velocities"
mkdir -p "${output}"

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python collect.py -i "G ${G}/${x} ${y}" -Ps $x --velocities -o "${output}/v_hist ${x} ${y}.txt"
