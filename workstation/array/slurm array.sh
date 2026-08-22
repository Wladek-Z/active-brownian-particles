#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 2G
#SBATCH --time 12:00:00
#SBATCH --job-name ABP
#SBATCH --array=1-256
#
#######################################

G=0

x=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Ps_params.txt)
y=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" Pf_params.txt)

echo "Task $SLURM_ARRAY_TASK_ID: Ps=$x Pf=$y"

mkdir -p "/data/biophys/ABP_channel/G ${G}/${x} ${y}"

source ../.venv/bin/activate
python abp_log.py -F "/data/biophys/ABP_channel/G ${G}/${x} ${y}" -tc timechain10000000.txt -Ps $x -Pf $y -G $G -N 1000
