#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=v100:1

echo "activating env"
source $HOME/projects/def-gberseth/$USER/RL4Chem/env_chem/bin/activate

echo "moving code to slurm tmpdir"
rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem

cd $SLURM_TMPDIR/RL4Chem

wandb offline
python train_reinforce_fc_agent.py target=drd2 seed=1 wandb_log=True wandb_dir='.' wandb_run_name='sdsd_'${s} &

python train_reinforce_fc_agent.py target=drd2 seed=1 rep=selfies wandb_log=True wandb_run_name='sdsd_'${s}

a="local_exp"
mkdir -p $HOME/projects/def-gberseth/$USER/RL4Chem/$a
cp -r $SLURM_TMPDIR/RL4Chem/wandb $HOME/projects/def-gberseth/$USER/RL4Chem/$a
echo "done"
