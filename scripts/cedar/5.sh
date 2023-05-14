#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=1-9

targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

seeds=(1 2 3)

s=${seeds[$(((SLURM_ARRAY_TASK_ID-1) % 3))]}
echo ${s}

t=${targets[$(((SLURM_ARRAY_TASK_ID-1) / 3))]}
echo ${t}

echo "activating env"
source $HOME/projects/def-gberseth/$USER/RL4Chem/env_chem/bin/activate

echo "moving code to slurm tmpdir"
rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem

cd $SLURM_TMPDIR/RL4Chem

python train_reinforce_rnn_agent.py target=${t} seed=${s} kl_coef=0.0001 wandb_log=True wandb_run_name='kltune_0.0001_reinforce_char_trans_smiles_'${s}