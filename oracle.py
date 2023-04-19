import os
import tdc
import yaml 
import wandb
import numpy as np
from rdkit import Chem
from docking import DockingVina

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.target = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 5000
            self.freq_log = 100
        else:
            self.args = args
            self.num_sub_proc = args.num_sub_proc
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        self.invalid_count = 0
    
    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_target(self, cfg):
        docking_config = dict()
        if cfg.target == 'fa7':
            box_center = (10.131, 41.879, 32.097)
            box_size = (20.673, 20.198, 21.362)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
        elif self.target == 'parp1':
            box_center = (26.413, 11.282, 27.238)
            box_size = (18.521, 17.479, 19.995)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
        elif self.target == '5ht1b':
            box_center = (-26.602, 5.277, 17.898)
            box_size = (22.5, 22.5, 22.5)
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'
        else:
            raise NotImplementedError

        box_parameter = (box_center, box_size)
        docking_config['box_parameter'] = box_parameter
        docking_config['vina_program'] = cfg.vina_program
        docking_config['temp_dir'] = cfg.temp_dir
        docking_config['exhaustiveness'] = cfg.exhaustiveness
        docking_config['num_sub_proc'] = cfg.num_sub_proc
        docking_config['num_cpu_dock'] = cfg.num_cpu_dock
        docking_config['num_modes'] = cfg.num_modes
        docking_config['timeout_gen3d'] = cfg.timeout_gen3d
        docking_config['timeout_dock'] = cfg.timeout_dock

        self.target = DockingVina(docking_config)
    
    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')
        
        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
        
    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
        
        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)

        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f} | '
                f'invalid_cnt: {self.invalid_count:.3f}')

        if self.args.wandb_log: 
            wandb.log({
                "avg_top1": avg_top1, 
                "avg_top10": avg_top10, 
                "avg_top100": avg_top100, 
                "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
                "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
                "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
                "avg_sa": avg_sa,
                "diversity_top100": diversity_top100,
                "n_oracle": n_calls,
            })
    
    def __len__(self):
        return len(self.mol_buffer) 
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        assert type(smiles_lst) == list
        score_list = [None] * len(smiles_lst)
        new_smiles = []
        new_smiles_ptrs = []
        for i, smi in enumerate(smiles_lst):
            if smi in self.mol_buffer:
                score_list[i] = self.mol_buffer[smi][0]
            else:
                new_smiles.append((smi))
                new_smiles_ptrs.append((i))

        new_smiles_scores = self.target.predict(new_smiles)
        
        for smi, ptr, sc in zip(new_smiles, new_smiles_ptrs, new_smiles_scores):
            if sc == 99.0:
                self.invalid_count += 1
                sc = 0
            else:
                self.mol_buffer[smi] = [sc, len(self.mol_buffer)+1]        

            score_list[ptr] = sc       

            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                # self.save_result(self.task_label) 

        return score_list
            
    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls