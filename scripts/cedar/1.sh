#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100l:1
#SBATCH --array=1-5

array=(-1 10 20 30 40 48)

rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_rl4chem
echo "moved code to slurm tmpdir"

singularity exec --nv --home $SLURM_TMPDIR --env WANDB_API_KEY="7da29c1c6b185d3ab2d67e399f738f0f0831abfc" --env REQUESTS_CA_BUNDLE="/usr/local/lib/python3.9/site-packages/certifi/cacert.pem" $SCRATCH/rl4chem.sif bash -c "source activate rl4chem && cd RL4Chem &&\
python train.py max_selfie_length=${array[SLURM_ARRAY_TASK_ID]} wandb_log=True seed=1 wandb_run_name=max_len${array[SLURM_ARRAY_TASK_ID]}_ent0.1_seed1" 