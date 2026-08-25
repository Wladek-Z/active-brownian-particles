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
python abp_trap.py -Ps $x -Pf $y -G $G -ttd "G ${G} results/ttd3 ${x} ${y}.txt" -btd "G ${G} results/btd3 ${x} ${y}.txt" -c "/data/biophys/ABP_channel/G ${G}/${x} ${y}"
