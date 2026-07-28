#!/bin/bash

G=$1
x=$2
y=$3

#SBATCH --partition short
#SBATCH --mem-per-cpu 10M
#SBATCH --time 2:00:00
#SBATCH --job-name ABP
#
#######################################

mkdir -p "G ${G}/${x} ${y}"

source ../.venv/bin/activate
python abp_log.py -F "G ${G}/${x} ${y}" -tc timechain10000000.txt -Ps $x -Pf $y -G $G -N 100
