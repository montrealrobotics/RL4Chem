#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100l:1
#SBATCH --array=1-5

rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_rl4chem

cd $SLURM_TMPDIR/RL4Chem

python train.py max_selfie_length=20 entropy_coefficient=0.01 wandb_log=True seed=$SLURM_ARRAY_TASK_ID wandb_run_name=max_len20_ent0.01_seed$SLURM_ARRAY_TASK_ID