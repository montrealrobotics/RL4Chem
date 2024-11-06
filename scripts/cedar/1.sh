#!/bin/bash

#SBATCH --account=def-gberseth
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=1-115

targets=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery'\
        'troglitazone_rediscovery'\
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity'\
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo'\
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo'\
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

seeds=(1 2 3 4 5)

t=${targets[$(((SLURM_ARRAY_TASK_ID-1) % 23))]}
echo ${t}

s=${seeds[$(((SLURM_ARRAY_TASK_ID-1) / 23))]}
echo ${s}

echo "activating env"
source $HOME/projects/def-gberseth/$USER/RL4Chem/env_chem/bin/activate

echo "moving code to slurm tmpdir"
rsync -a $HOME/projects/def-gberseth/$USER/RL4Chem/ $SLURM_TMPDIR/RL4Chem --exclude=env_chem

cd $SLURM_TMPDIR/RL4Chem

python train_reinforce_trans_agent.py dataset=molgen_oled_1 target=qed seed=${s} wandb_log=True wandb_run_name='reb_alpha_1'${s}
