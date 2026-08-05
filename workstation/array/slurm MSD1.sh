#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

G=1
offset=0
output="G ${G} results/MSD o${offset}"

x=$1
y=$2

mkdir -p "${output}"

echo "Task $SLURM_JOB_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python 'get stats.py' --MSD -F "G ${G}/${x} ${y}" -Ps $x -Pf $y -o "${output}" -off ${offset} -tc timechain10000000.txt -f "lags o${offset}.npz"
