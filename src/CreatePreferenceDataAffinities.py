# %%

# %%
import numpy as np

# sampled_smiles = np.array(sampled_smiles)


# %%

import dockstring
from dockstring import load_target

sampled_affinities =[]
target = load_target('LCK') 

# %%
config = {
    'batch_size' :512,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 1024,
    'lr': 1e-6,
    'epochs': 2,
    'properties': sorted(['affinity', 'qeds', 'logps','tpsas', 'sas'])
}
config['run_name'] = "LCK_DOCKSTRING_"+ "_".join(prop for prop in config['properties'])
print(config)

import pickle
with open('../checkpoints/'+ config['run_name'] +'/RawPreferenceData.pkl', 'rb') as f:
    target_smiles, target_properties, sampled_smiles = pickle.load(f)


print(len(target_smiles), len(target_properties), len(sampled_smiles))

# %%
import concurrent.futures

from rdkit.Chem import RDConfig
import os
import sys
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
# import sascorer
from rdkit.Chem import QED, Descriptors, Crippen

def dock_smile(smile):
    target = load_target('LCK')
    try:
        score, _ =  target.dock(smile, num_cpus=10)
    except:
        score = None
    return score



data = []
import multiprocessing as mp

for target_smile, target_prop, smilelist in zip(target_smiles, target_properties, sampled_smiles):
    
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(dock_smile, smilelist)
    
    data.append([target_smile, target_prop, smilelist, results])
    
    if len(data) % 100 == 0:
        with open('../checkpoints/'+config ['run_name'] +'/PreferenceDataAffinities.pkl', 'wb') as f:
            pickle.dump(data, f)
            
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities.pkl', 'wb') as f:
    pickle.dump(data, f)

# %%



