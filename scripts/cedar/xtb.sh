#!/bin/bash

#SBATCH --account=def-gberseth
#SBATCH --time=00:45:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=4G

python props/xtb/stda_xtb.py