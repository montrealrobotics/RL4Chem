#!/bin/bash

#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --partition=main
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --array=1-3

array=(-1 "0.5" "10" "100")

module --quiet load anaconda/3
conda activate rl4chem
echo "activated conda environment"

rsync -a $HOME/RL4Chem/ $SLURM_TMPDIR/RL4Chem
echo "moved code to slurm tmpdir"

cd $SLURM_TMPDIR/RL4Chem/regression
python regression.py seed=$SLURM_ARRAY_TASK_ID wandb_log=True wandb_run_name="selfies_charnn_"$SLURM_ARRAY_TASK_ID