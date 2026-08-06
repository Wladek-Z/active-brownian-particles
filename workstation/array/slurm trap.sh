#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-256
#
#######################################

G=1

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

mkdir -p "G ${G} results/ttd data"
mkdir -p "G ${G} results/btd data"

source ../.venv/bin/activate
python abp_trap.py -Ps $x -Pf $y -G $G -ttd "G ${G} results/ttd data/${x} ${y}.txt" -btd "G ${G} results/btd data/${x} ${y}.txt"
