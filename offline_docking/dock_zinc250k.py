import time
import wandb
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from docking import DockingVina

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str, default='raj19', help="wandb entity")
    parser.add_argument("--wandb-run-name", type=str, default='zinc250', help="wandb run name")
    parser.add_argument("--num-sub-proc", type=int, default=12, help="number of sub processes")
    parser.add_argument("--num-parallel", type=int, default=12, help="number of parallel evaluations")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    with wandb.init(project='offline_docking', entity=args.wandb_entity, config=args.__dict__):
        wandb.run.name = args.wandb_run_name

        #create folder for saving docking scores
        Path("docked_data/").mkdir(parents=True, exist_ok=True)

        #loading dataset
        zinc_df = pd.read_csv('filtered_data/zinc250_selfies_2.1.0_.csv')
        zinc_df_canon_smiles = zinc_df.canon_smiles
        num_smiles = len(zinc_df_canon_smiles)

        #Setting up docking software
        target = 'fa7'
        docking_config = dict()

        if target == 'fa7':
            box_center = (10.131, 41.879, 32.097)
            box_size = (20.673, 20.198, 21.362)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
        elif target == 'parp1':
            box_center = (26.413, 11.282, 27.238)
            box_size = (18.521, 17.479, 19.995)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
        elif target == '5ht1b':
            box_center = (-26.602, 5.277, 17.898)
            box_size = (22.5, 22.5, 22.5)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'

        box_parameter = (box_center, box_size)
        docking_config['vina_program'] = 'qvina2'
        docking_config['box_parameter'] = box_parameter
        docking_config['temp_dir'] = 'tmp'
        docking_config['exhaustiveness'] = 1
        docking_config['num_sub_proc'] = args.num_sub_proc
        docking_config['num_cpu_dock'] = 1
        docking_config['num_modes'] = 10 
        docking_config['timeout_gen3d'] = 30
        docking_config['timeout_dock'] = 100 

        predictor = DockingVina(docking_config)
        docking_scores = np.array([0.0 for i in range(num_smiles)])
        zinc_df.insert(0, target, docking_scores, allow_duplicates=False)

        metrics = dict()
        STEP = args.num_parallel
        for i in range(0, num_smiles, STEP):
            current_smiles_list = zinc_df_canon_smiles[i:i+STEP]
            
            reward_start_time = time.time()
            current_scores_list = predictor.predict(current_smiles_list)
            reward_eval_time = time.time() - reward_start_time

            metrics['reward_eval_time'] = reward_eval_time
            metrics['max eval score'] = np.max(current_scores_list)
            metrics['min eval score'] = np.min(current_scores_list)
            metrics['mean eval score'] = np.mean(current_scores_list)
            wandb.log(metrics)
            
            docking_scores[i:i+STEP] = current_scores_list
            
            print("Total molecules docked: {}, average evaluation time: {}".format(i, reward_eval_time))

            if i%50000 == 0:
                #saving docked values in csv file
                zinc_df[target] = docking_scores                
                zinc_df.to_csv('docked_data/zinc250_selfies_docked_.csv')

        #saving docked values in csv file
        zinc_df[target] = docking_scores
        zinc_df.to_csv('docked_data/zinc250_selfies_docked_.csv')