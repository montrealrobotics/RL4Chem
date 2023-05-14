targets=('fa7')

for t in ${targets[@]}; 
do
    python ../../train_reinforce_trans_agent.py target=$t seed=1 task=augmented_docking num_sub_proc=24 dataset=zinc1m wandb_log=True wandb_run_name='reinforce_zinc1m_1'
done

# for t in ${targets[@]}; 
# do
#     python ../../train_reinforce_trans_agent.py target=$t seed=2 task=augmented_docking num_sub_proc=24 dataset=zinc1m wandb_log=True wandb_run_name='reinforce_zinc1m_2'
# done

# for t in ${targets[@]}; 
# do
#     python ../../train_reinforce_trans_agent.py target=$t seed=3 task=augmented_docking num_sub_proc=24 dataset=zinc1m wandb_log=True wandb_run_name='reinforce_zinc1m_3'
# done
