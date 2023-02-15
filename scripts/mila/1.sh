#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -c 24
#SBATCH --partition=real-lab
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --array=1-3

array=("fa7" "parp1" "5ht1b")

module --quiet load anaconda/3
conda activate rl4chem
echo "activated conda environment"

rsync -a $HOME/RL4Chem/ $SLURM_TMPDIR/RL4Chem
echo "moved code to slurm tmpdir"

python train.py target=${array[SLURM_ARRAY_TASK_ID]} num_sub_proc=24 wandb_log=True wandb_run_name=max_len_25
