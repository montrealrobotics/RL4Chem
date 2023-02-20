#!/bin/bash

#SBATCH --account=rrg-gberseth
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100l:1
#SBATCH --array=1-3

array=(-1 "fa7" "parp1" "5ht1b")

rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_rl4chem
echo "moved code to slurm tmpdir"

singularity exec --nv --home $SLURM_TMPDIR --env WANDB_API_KEY="7da29c1c6b185d3ab2d67e399f738f0f0831abfc",REQUESTS_CA_BUNDLE="/usr/local/envs/rl4chem/lib/python3.11/site-packages/certifi/cacert.pem",HYDRA_FULL_ERROR=1 $SCRATCH/rl4chem.sif bash -c "source activate rl4chem && cd RL4Chem &&\
python train.py target=${array[SLURM_ARRAY_TASK_ID]} wandb_log=True wandb_run_name=onehotconv_max_len_25_1 seed=1 num_sub_proc=20" 