#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100l:1
#SBATCH --array=1-3

array=(-1 "0.5" "10" "100")

rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_rl4chem
echo "moved code to slurm tmpdir"

singularity exec --nv --home $SLURM_TMPDIR --env WANDB_API_KEY="7da29c1c6b185d3ab2d67e399f738f0f0831abfc",REQUESTS_CA_BUNDLE="/usr/local/envs/rl4chem/lib/python3.11/site-packages/certifi/cacert.pem",HYDRA_FULL_ERROR=1 $SCRATCH/rl4chem_old.sif bash -c "source activate rl4chem && cd RL4Chem && pip install scikit-learn &&\
python regression.py max_grad_norm=0.5 seed=1 wandb_log=True wandb_run_name=0.5_charnn_$SLURM_ARRAY_TASK_ID"