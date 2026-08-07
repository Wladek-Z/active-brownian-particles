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

mkdir -p "G ${G} results/ttd data"
mkdir -p "G ${G} results/btd data"

source ../.venv/bin/activate
python abp_trap.py -Ps $x -Pf $y -G $G -ttd "G ${G} results/ttd data/${x} ${y}.txt" -btd "G ${G} results/btd data/${x} ${y}.txt" -c "G ${G}/${x} ${y}"
