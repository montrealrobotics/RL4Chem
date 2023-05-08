#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=v100:1

module load httpproxy
echo "activating env"
source $HOME/projects/def-gberseth/$USER/RL4Chem/env_chem/bin/activate

echo "moving code to slurm tmpdir"
rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem

cd $SLURM_TMPDIR/RL4Chem

python train_reinforce_fc_agent.py target=drd2 seed=1 wandb_log=True wandb_run_name='sdsd_'${s} &

python train_reinforce_fc_agent.py target=drd2 seed=1 rep=selfies wandb_log=True wandb_run_name='sdsd_'${s}

echo "done"