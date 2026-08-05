#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 2G
#SBATCH --time 6:00:00
#SBATCH --job-name ABP
#
#######################################

G=1

x=$1
y=$2

echo "Task $SLURM_JOB_ID: Ps=$x Pf=$y"

mkdir -p "G ${G}/${x} ${y}"

source ../.venv/bin/activate
python abp_log.py -F "G ${G}/${x} ${y}" -tc timechain10000000.txt -Ps $x -Pf $y -G $G -N 1000
