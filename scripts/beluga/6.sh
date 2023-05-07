#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=v100:1
#SBATCH --array=1-9

targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

seeds=(1 2 3)

s=${seeds[$(((SLURM_ARRAY_TASK_ID-1) % 3))]}
echo ${s}

t=${targets[$(((SLURM_ARRAY_TASK_ID-1) / 3))]}
echo ${t}

rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem
echo "moved code to slurm tmpdir"

cd $SLURM_TMPDIR/RL4Chem

wandb offline
python train_reinforce_trans_agent.py target=${t} seed=${s} learning_rate=0.0001 wandb_log=True wandb_dir='./' wandb_run_name='lr_0.0001_reinforce_char_trans_smiles_'${s}

a="local_exp"
mkdir -p $HOME/projects/def-gberseth/$USER/RL4Chem/$a
cp -r $SLURM_TMPDIR/RL4Chem/wandb $HOME/projects/def-gberseth/$USER/RL4Chem/$a