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

output="G ${G} results/MSD ind"

mkdir -p "${output}"

source ../.venv/bin/activate
python 'get stats.py' --ppMSD -F "G ${G}/${x} ${y}" -Ps $x -Pf $y -o "${output}" -f "lags o0.npz" -s 20
