targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

seeds=(1 2 3)

for t in ${targets[@]}; 
do
for s in ${seeds[@]}; 
do
echo running target = $t, seed = $s
cd ../../
python train_reinforce_trans_agent.py target=${t} seed=${s} #wandb_log=True wandb_run_name='reinforce_char_rnn_smiles_'${s}

done
done