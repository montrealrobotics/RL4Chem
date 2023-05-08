targets=('troglitazone_rediscovery' 'sitagliptin_mpo' 'median2')

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=1 rep=selfies wandb_log=True wandb_run_name='std_selfies_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=2 rep=selfies wandb_log=True wandb_run_name='std_selfies_1'
done

for t in ${targets[@]}; 
do
    python ../../train_reinforce_rnn_agent.py oracle=$t seed=3 rep=selfies wandb_log=True wandb_run_name='std_selfies_1'
done
