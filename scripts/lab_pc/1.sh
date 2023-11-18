targets=('drd2' 'qed' 'jnk3' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery' 'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' 'isomers_c7h8n2o2' 'sitagliptin_mpo')        

for t in ${targets[@]}; 
do
    python ../../train_reinvent_replay_agent.py target=$t seed=1 wandb_log=True wandb_run_name='table'
done
