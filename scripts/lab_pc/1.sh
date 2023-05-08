targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py target=$t seed=1 rep=selfies penalty=len wandb_log=True wandb_run_name='len_selfies_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py target=$t seed=2 rep=selfies penalty=len wandb_log=True wandb_run_name='len_selfies_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py target=$t seed=3 rep=selfies penalty=len wandb_log=True wandb_run_name='len_selfies_1'
done
