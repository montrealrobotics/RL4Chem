targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=1 wandb_log=True wandb_run_name='std_smiles_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=2 wandb_log=True wandb_run_name='std_smiles_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=3 wandb_log=True wandb_run_name='std_smiles_1'
done
