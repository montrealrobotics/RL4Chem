import os
import tdc
import yaml
import wandb
import numpy as np
from tdc import Oracle
from rdkit import Chem

def top_auc(buffer, top_n, finish, env_log_interval, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(env_log_interval, min(len(buffer), max_oracle_calls), env_log_interval):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += env_log_interval * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

class BaseOptimizer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.target_name = cfg.target
        
        # defining target oracles
        self.assign_target(cfg)

        # defining standard oracles
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

        self.max_oracle_calls = cfg.max_oracle_calls
        self.env_log_interval = cfg.env_log_interval
        
        # store all unique molecules
        self.mol_buffer = dict()

        self.mean_score = 0

        #logging counters
        self.last_log = 0
        self.total_count = 0
        self.invalid_count = 0
        self.redundant_count = 0
        
    @property
    def budget(self):
        return self.max_oracle_calls
    
    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls
    
    def assign_target(self, cfg):
        if cfg.task == 'docking':
            from docking import DockingVina
            docking_config = dict()
            if self.target_name == 'fa7':
                box_center = (10.131, 41.879, 32.097)
                box_size = (20.673, 20.198, 21.362)
            elif self.target_name == 'parp1':
                box_center = (26.413, 11.282, 27.238)
                box_size = (18.521, 17.479, 19.995)
            elif self.target_name == '5ht1b':
                box_center = (-26.602, 5.277, 17.898)
                box_size = (22.5, 22.5, 22.5)
            elif self.target_name == 'jak2':
                box_center = (114.758,65.496,11.345)
                box_size= (19.033,17.929,20.283)
            elif self.target_name == 'braf':
                box_center = (84.194,6.949,-7.081)
                box_size = (22.032,19.211,14.106)
            else:
                raise NotImplementedError
            
            docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/' + self.target_name + '/receptor.pdbqt'
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
            self.predict = self.predict_docking

        elif cfg.task == 'pmo':
            self.target = Oracle(name = self.target_name)
            self.predict = self.predict_pmo
        else:
            raise NotImplementedError
    
    def define_wandb_metrics(self):
        #new wandb metric
        wandb.define_metric("num_molecules")
        wandb.define_metric("avg_top1", step_metric="num_molecules")
        wandb.define_metric("avg_top10", step_metric="num_molecules")
        wandb.define_metric("avg_top100", step_metric="num_molecules")
        wandb.define_metric("auc_top1", step_metric="num_molecules")
        wandb.define_metric("auc_top10", step_metric="num_molecules")
        wandb.define_metric("auc_top100", step_metric="num_molecules")
        wandb.define_metric("avg_sa", step_metric="num_molecules")
        wandb.define_metric("diversity_top100", step_metric="num_molecules")
        wandb.define_metric("n_oracle", step_metric="num_molecules")
        wandb.define_metric("invalid_count", step_metric="num_molecules")
        wandb.define_metric("redundant_count", step_metric="num_molecules")

    def score_pmo(self, smi):
        """
        Function to score one molecule
        Argguments:
            smi: One SMILES string represnets a moelcule.
        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            self.invalid_count += 1
            return 0.0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                self.mol_buffer[smi][2] += 1
                self.redundant_count += 1
            else:
                self.mol_buffer[smi] = [float(self.target(smi)), len(self.mol_buffer)+1, 1]
            return self.mol_buffer[smi][0]
        
    def predict_pmo(self, smiles_list):
        assert type(smiles_list) == list
        self.total_count += len(smiles_list)
        score_list = []
        for smi in smiles_list:
            score_list.append(self.score_pmo(smi))
            if len(self.mol_buffer) % self.env_log_interval == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
        self.mean_score = np.mean(score_list)
        return score_list

    def predict_docking(self, smiles_list):
        """
        Score
        """
        assert type(smiles_list) == list
        self.total_count += len(smiles_list)
        score_list = [None] * len(smiles_list)
        new_smiles = []
        new_smiles_ptrs = []
        for i, smi in enumerate(smiles_list):
            if smi in self.mol_buffer:
                score_list[i] = self.mol_buffer[smi][0] / 20
                self.mol_buffer[smi][2] += 1
                self.redundant_count += 1
            else:
                new_smiles.append((smi))
                new_smiles_ptrs.append((i))

        new_smiles_scores = self.target(new_smiles)

        for smi, ptr, sc in zip(new_smiles, new_smiles_ptrs, new_smiles_scores):
            if sc == 99.0:
                self.invalid_count += 1
                sc = 0
            else:
                self.mol_buffer[smi] = [-sc, len(self.mol_buffer)+1, 1]        

            score_list[ptr] = -sc / 20      

            if len(self.mol_buffer) % self.env_log_interval == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
        self.mean_score = np.mean(score_list)
        return score_list

    def optimize(self, cfg):
        raise NotImplementedError

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
            
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
                # f'avg_top1: {avg_top1:.3f} | '
                # f'avg_top10: {avg_top10:.3f} | '
                # f'avg_top100: {avg_top100:.3f} | '
                f'mean_score: {self.mean_score:.3f} | '
                # f'avg_sa: {avg_sa:.3f} | '
                # f'div: {diversity_top100:.3f} | '
                f'tot_cnt: {self.total_count} | '
                f'inv_count: {self.invalid_count} | '
                f'red_cnt: {self.redundant_count} | '
                )

        if self.cfg.wandb_log: 
            wandb.log({
                "avg_top1": avg_top1, 
                "avg_top10": avg_top10, 
                "avg_top100": avg_top100, 
                "auc_top1": top_auc(self.mol_buffer, 1, finish, self.env_log_interval, self.max_oracle_calls),
                "auc_top10": top_auc(self.mol_buffer, 10, finish, self.env_log_interval, self.max_oracle_calls),
                "auc_top100": top_auc(self.mol_buffer, 100, finish, self.env_log_interval, self.max_oracle_calls),
                "avg_sa": avg_sa,
                "diversity_top100": diversity_top100,
                "invalid_count" : self.invalid_count,
                "redundant_count": self.redundant_count,
                "num_molecules": n_calls,
            })
    
    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores), 
                np.mean(scores[:10]), 
                np.max(scores), 
                self.diversity_evaluator(smis), 
                np.mean(self.sa_scorer(smis)), 
                float(len(smis_pass) / 100), 
                top1_pass]
    
    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.cfg.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.cfg.output_dir, 'results_' + suffix + '.yaml')
        
        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def __len__(self):
        return len(self.mol_buffer)