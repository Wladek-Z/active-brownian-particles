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

echo "Task $SLURM_JOB_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python collect.py -i "G ${G}/${x} ${y}" -Ps $x --trajectory -o "trajs ${x} ${y}.npz" -Pf $y -s 20
