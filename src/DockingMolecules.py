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
mols_df = pd.read_csv('../checkpoints/DPO_Finetuning_LCK_Decoder_affinity_logps_qeds_tpsas/sampled_mols.csv')
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

mols_df['input_affinity'] = affinity_scaler.inverse_transform(mols_df['input_affinity'].values.reshape(-1,1))
mols_df['input_logps'] = logp_scaler.inverse_transform(mols_df['input_logps'].values.reshape(-1,1))
mols_df['input_tpsas'] = tpsas_scaler.inverse_transform(mols_df['input_tpsas'].values.reshape(-1,1))
# mols_df['input_qeds'] = qed_scaler.inverse_transform(mols_df['input_qeds'].values.reshape(-1,1))

# %%
mols_df['computed_affinity'] = [None]*len(mols_df)
mols_df['computed_logps'] = [None]*len(mols_df)
mols_df['computed_tpsas'] = [None]*len(mols_df)
# mols_df['computed_qeds'] = [None]*len(mols_df)

# %%
from tqdm.auto import tqdm

target = load_target('PPARD')
for i, row in tqdm(mols_df.iterrows()):
    smi = row['SMILES']
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        try:
            
            # qed = QED.qed(mol)
            logp = Crippen.MolLogP(mol)
            mols_df.at[i, 'computed_logps'] = logp
            
            tpsa = Descriptors.TPSA(mol)
            mols_df.at[i, 'computed_tpsas'] = tpsa
            
            score, info = target.dock(smi, num_cpus=10)
            mols_df.at[i, 'computed_affinity'] = score
        except:
            pass
        
    if i % 100 == 0:
        # mols_df.dropna(inplace=True)
        mols_df.to_csv('../checkpoints/DPO_Finetuning_LCK_Decoder_affinity_logps_qeds_tpsas/temp.csv', index=False)
    # break

# %%


# %%


# %%
# mols_df = pd.read_csv('../checkpoints/Decoder_affinity_logps_tpsas/temp.csv')
# mols_df.dropna(inplace=True)

# # %%
# import seaborn as sns
# sns.pairplot(mols_df[['input_affinity', 
#                       'input_logps', 
#                       'input_tpsas', 
#                       'computed_affinity', 
#                       'computed_logps', 
#                       'computed_tpsas']])

# %%



