import numpy as np
import selfies as sf
from docking import DockingVina

if __name__ == '__main__':
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
    docking_config['num_sub_proc'] = 10
    docking_config['num_cpu_dock'] = 5
    docking_config['num_modes'] = 10 
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100 

    predictor = DockingVina(docking_config)

    selfie_alphabet_set = sf.get_semantic_robust_alphabet()
    smiles_benzene = "c1ccccc1"
    selfies_benzene = sf.encoder(smiles_benzene)
    selfie_alphabet_set.add(selfies_benzene)
    selfie_alphabets = list(selfie_alphabet_set)

    smiles_list = []
    for i in range(25):

        selfie_str=''.join(np.random.choice(selfie_alphabets, 4))
        decoded_smile_str = sf.decoder(selfie_str)
        smiles_list.append(decoded_smile_str)

    a_list = predictor.predict(smiles_list)
    print(a_list)