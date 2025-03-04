# %%
import numpy as np
import pandas as pd

import dockstring
from dockstring import load_target

import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem as Chem


from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit.Chem import QED, Descriptors, Crippen

# %%
# BETA = 0.9
# config = {
#     'batch_size' :512,
#     'd_model': 512,
#     'n_heads': 8,
#     'n_layers':6,
#     'hidden_units': 1024,
#     'lr': 1e-6,
#     'epochs': 2,
#     'properties': sorted(['affinity', 'logps', 'qeds', 'tpsas', 'sas'])
# }
# config['run_name'] = "LCK_DOCKSTRING_"+ "_".join(prop for prop in config['properties'])

# config['run_name'] = "LCK_Decoder_"+ "_".join(prop for prop in config['properties'])
# print(config)

# config['beta'] = BETA
# config['epochs'] = 30

config = {
    'batch_size' :512,
    'd_model': 512,
    'n_heads': 8,
    'n_layers':6,
    'hidden_units': 1024,
    'lr': 1e-6,
    'epochs': 2,
    'properties': sorted(['affinity', 'sas', 'logps', 'tpsas', 'qeds']),
}
config['run_name'] = "LCK_DOCKSTRING_"+ "_".join(prop for prop in config['properties'])

config['run_name']= 'IPO_Finetuning_BETA_' + str(sys.argv[1]) + '_' + config['run_name']


# %%
import sys


# %%
df = pd.read_csv('../data/DOCKSTRING/target_specific/LCK.csv')

import sklearn
from sklearn.preprocessing import MinMaxScaler

affinity_scaler = MinMaxScaler()
qed_scaler = MinMaxScaler()
logp_scaler = MinMaxScaler()
tpsas_scaler = MinMaxScaler()

affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
qed_scaler.fit(df['qeds'].values.reshape(-1,1))
logp_scaler.fit(df['logps'].values.reshape(-1,1))
tpsas_scaler.fit(df['tpsas'].values.reshape(-1,1))



# %%
import pickle 

with open('../checkpoints/' + config['run_name'] + '/results.pkl', 'rb') as f:
    results = pickle.load(f)

# %%
print(results.keys())
print(len(results.keys()))

# %%

# logps, tpsas, qeds, samples

# %%
import multiprocessing as mp
import pandas as pd

def dock_smile(smile):
    target = load_target('LCK')
    try:
        score, _ =  target.dock(smile, num_cpus=2)
    except:
        score = None
    return score

# %%
K = 16

data = {}
for key in results:
    print("=========================================================")
    DIST = results[key][-1]
    DIST = np.array(DIST)
    DIST = DIST.reshape(K, int(len(DIST)/K))

    all_scores = []
    for sample_list in DIST:
        with mp.Pool(mp.cpu_count()) as pool:
            scores = pool.map(dock_smile, sample_list)
        all_scores += scores 
    all_scores = np.array(all_scores)
    all_scores = all_scores[~pd.isna(all_scores)]
    data[key] = all_scores
    
    pickle.dump(data, open('../checkpoints/' + config['run_name'] + '/affinity_distribution.pkl', 'wb'))
    
for key in data:
    plt.hist(data[key], bins=10, alpha=0.5, label="-" + key.split('-')[1])
    
plt.legend()
plt.savefig('../checkpoints/' + config['run_name'] + '/affinity_distribution.png')

# %%
#len(scores)

# %%
# for i in [1,0.5,0.7]:
#     plt.hist(scores*i, bins=10, alpha=0.5, label="*"+str(i))
# plt.legend()
# plt.savefig('../checkpoints/' + config['run_name'] + '/affinity_distribution.png')


# %%



