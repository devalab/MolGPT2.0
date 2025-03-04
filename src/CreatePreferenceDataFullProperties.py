# %%
import multiprocessing as mp


def dock_smile(smile):
    target = load_target('LCK')
    try:
        score, _ =  target.dock(smile)
    except:
        score = None
    return score

# with mp.Pool(mp.cpu_count()) as p:
#     results = p.map(dock_smile, smilelist)

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

# %%
import pickle
with open('../checkpoints/'+ config['run_name'] +'/PreferenceDataAffinities.pkl', 'rb') as f:
    data = pickle.load(f)

# %%
print(len(data))

# %%

import rdkit
import rdkit.Chem as Chem

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit.Chem import QED, Descriptors, Crippen

def compute_logp(mol):
    try:
        logp = Crippen.MolLogP(mol)
    except:
        logp = None
    return logp

def compute_qed(mol):
    try:
        qed = QED.qed(mol)
    except:
        qed = None
    return qed

def compute_tpsa(mol):
    try:
        tpsa = Descriptors.TPSA(mol)
    except:
        tpsa = None
    return tpsa

def compute_sas(mol):
    try:
        sas = sascorer.calculateScore(mol)
    except:
        sas = None
    return sas

def compute_props(smile):
    # try:
    #     mol = Chem.MolFromSmiles(smile)
    # except:
    #     return [None, None, None]
    mol = Chem.MolFromSmiles(smile)
    logp = compute_logp(mol)
    qed = compute_qed(mol)
    tpsa = compute_tpsa(mol)
    # sas = compute_sas(mol)
    return [logp, qed, tpsa]


# %%
print(len(data))

# %%
print(data[0])

# %%


# %%
all_data = []
for i, row in enumerate(data):
    print(" iter ",i)
    target_smile = row[0]
    target_properties = row[1]
    candidate_smiles = row[2]
    candidate_affinities = row[3]
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(compute_props, candidate_smiles)
    all_data.append([target_smile,
                    target_properties,
                    candidate_smiles,
                    candidate_affinities,
                    results])
    if len(all_data) % 100 == 0:
        with open('../checkpoints/'+config['run_name']+'/PreferenceDataFullProperties.pkl', 'wb') as f:
            pickle.dump(all_data, f)

# %%

import pickle
with open('../checkpoints/'+config['run_name'] +'/PreferenceDataFullProperties.pkl', 'rb') as f:
    all_data = pickle.load(f)

# %%
len(all_data)

# %%
all_data[0]

# %%
import pandas as pd
df = pd.read_csv('../data/DOCKSTRING/target_specific/LCK.csv')

# %%
import sklearn
# minmaxscaler
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df['qeds'] = scaler.fit_transform(df['qeds'].values.reshape(-1,1))
# df['tpsas'] = scaler.fit_transform(df['tpsas'].values.reshape(-1,1))
# df['logps'] = scaler.fit_transform(df['logps'].values.reshape(-1,1))
# df['affinity'] = scaler.fit_transform(df['affinity'].values.reshape(-1,1))

affinity_scaler = MinMaxScaler()
qed_scaler = MinMaxScaler()
logp_scaler = MinMaxScaler()
tpsas_scaler = MinMaxScaler()

affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
qed_scaler.fit(df['qeds'].values.reshape(-1,1))
logp_scaler.fit(df['logps'].values.reshape(-1,1))
tpsas_scaler.fit(df['tpsas'].values.reshape(-1,1))

df['qeds'] = qed_scaler.transform(df['qeds'].values.reshape(-1,1))
df['logps'] = logp_scaler.transform(df['logps'].values.reshape(-1,1))
df['tpsas'] = tpsas_scaler.transform(df['tpsas'].values.reshape(-1,1))
df['affinity'] = affinity_scaler.transform(df['affinity'].values.reshape(-1,1))


# %%
def compute_preference_score(target_properties, candidate_properties):
    # compute the preference score
    # target_properties = [aff, logp, qed, tpsa]
    # candidate_properties = [aff, logp, qed, tpsa]
    # preference_score = 1 - (1/4) * (abs(aff - aff') + abs(logp - logp') + abs(qed - qed') + abs(tpsa - tpsa'))
    return 1 - (1/4) * np.sum(np.abs(target_properties - candidate_properties))
    # print(np.abs(target_properties[0] - candidate_properties[0]), 1 - np.abs(target_properties[0] - candidate_properties[0]))
    # return 1 - np.abs(target_properties[0] - candidate_properties[0])
    # return 1 - (1/2) * (np.abs(target_properties[0] - candidate_properties[0]) + np.abs(target_properties[1] - candidate_properties[1]))

import numpy as np
PreferenceData = []
for i, row in enumerate(all_data):
    target_smile = row[0]
    target_properties = np.array(row[1])
    candidate_smiles = np.array(row[2])
    candidate_affinities = np.array(row[3])
    candidate_properties = np.array(row[4])
    
    tuples = []
    for smi, aff, prop in zip(candidate_smiles, candidate_affinities, candidate_properties):
        tuples.append([aff] + list(prop)) 
    tuples = np.array(tuples)
    
    scaled_affs = affinity_scaler.transform(tuples[:,0].reshape(-1,1))
    scaled_logps = logp_scaler.transform(tuples[:,1].reshape(-1,1))
    scaled_qeds = qed_scaler.transform(tuples[:,2].reshape(-1,1))
    scaled_tpsas = tpsas_scaler.transform(tuples[:,3].reshape(-1,1))
   
    tuples = np.concatenate([scaled_affs, scaled_logps, scaled_qeds, scaled_tpsas], axis=1)
    
    preference_scores = []
    for i in range(len(tuples)):
        score = compute_preference_score(target_properties, tuples[i])
        if not np.isnan(score):
            # print(score)
            preference_scores.append([ candidate_smiles[i], score, tuples[i]])
    
    preference_scores = sorted(preference_scores, key=lambda x: x[1], reverse=True)
    good_sample = preference_scores[0]
    bad_sample = preference_scores[-1]
    PreferenceData.append([target_smile, target_properties, good_sample, bad_sample])
    
    
with open('../checkpoints/'+config['run_name']+'/PreferenceData.pkl', 'wb') as f:
    pickle.dump(PreferenceData, f)

# %%
print(len(PreferenceData))

# %%
PreferenceData[0]

# %%
PreferenceData[0]

# %%



