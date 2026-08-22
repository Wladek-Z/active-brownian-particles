#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

G=0
offset=0
output="G ${G} NV results/MSD ind"

x=$1
y=$2

mkdir -p "${output}"

echo "Task $SLURM_JOB_ID: Ps=$x Pf=$y"

source ../.venv/bin/activate
python 'get stats.py' --ppMSD -s 20 -F "/data/biophys/ABP_channel/G ${G} NV/${x} ${y}" -o "${output}/${x} ${y}" -off ${offset} -f "lags o${offset}.npz"
