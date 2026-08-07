#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

G=1

x=$1
y=$2

output="G ${G} results/x trajectories"
mkdir -p "${output}"

echo "Task $SLURM_JOB_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python collect.py -i "G ${G}/${x} ${y}" -Ps $x --trajectory -o "${output}/trajs ${x} ${y}.npz" -s 20 -Pf $y
