#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=00:45:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=1-11

targets=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery'\
        'troglitazone_rediscovery' 'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' 'isomers_c7h8n2o2' 'sitagliptin_mpo')        

seeds=(1)

t=${targets[$(((SLURM_ARRAY_TASK_ID-1) % 23))]}
echo ${t}

s=${seeds[$(((SLURM_ARRAY_TASK_ID-1) / 23))]}
echo ${s}

echo "activating env"
source $HOME/projects/def-gberseth/$USER/RL4Chem/env_chem/bin/activate

echo "moving code to slurm tmpdir"
rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem

cd $SLURM_TMPDIR/RL4Chem

python train_reinvent_replay_agent.py target=${t} seed=${s} wandb_log=True wandb_run_name='table'
