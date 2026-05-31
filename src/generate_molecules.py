import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
import itertools
from tqdm.auto import tqdm
import os

from build_corpus import build_corpus
from build_vocab import WordVocab
from utils import split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
import argparse
import sklearn
import wandb
from sklearn.preprocessing import MinMaxScaler
from rdkit import DataStructs

from train_models import CustomTargetDataset, PositionalEncodings, PropertyEncoder, set_up_causal_mask, MolGPT2, Sampler, compute_metrics, load_model, sample_a_bunch
print("Loaded libraries.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties to use (e.g., --properties affinity logps)')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory in which model is saved')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for loading test set')
parser.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--num_samples', type=int, default=128, help='Number of molecules to generate per target property combination')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units in feedforward layers')
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
parser.add_argument('--affinity_targets', nargs='+', type=float, default=[-10,-9,-8,-7,-6], help='Target affinity values for generation')
parser.add_argument('--logp_targets', nargs='+', type=float, default=[1,3,5], help='Target logP values for generation')
parser.add_argument('--qed_targets', nargs='+', type=float, default=[0.4,0.6,0.8], help='Target QED values for generation')
parser.add_argument('--tpsa_targets', nargs='+', type=float, default=[40,70,100], help='Target TPSA values for generation')
parser.add_argument('--sas_targets', nargs='+', type=float, default=[2,3,4], help='Target SAS values for generation')
args = parser.parse_args()
print("Properties to use: ", args.properties)

config = {
    'batch_size' : args.batch_size,
    'd_model': args.d_model,
    'n_heads': args.n_heads,
    'n_layers': args.n_layers,
    'hidden_units': args.hidden_units,
    'properties': sorted(args.properties),
    'lr' : args.lr,
}
if args.checkpoint_dir is not None:
    config['run_name'] = args.checkpoint_dir
else:
    config['run_name'] = "encoder_decoder_"+ "_".join(prop for prop in config['properties'])

df = pd.read_csv('../data/lck_dockstring_data1.csv')
print(df.head())

affinity_scaler = MinMaxScaler()
qed_scaler = MinMaxScaler()
logp_scaler = MinMaxScaler()
tpsas_scaler = MinMaxScaler()
sas_scaler = MinMaxScaler()

affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
qed_scaler.fit(df['qed'].values.reshape(-1,1))
logp_scaler.fit(df['logp'].values.reshape(-1,1))
tpsas_scaler.fit(df['tpsa'].values.reshape(-1,1))
sas_scaler.fit(df['sas'].values.reshape(-1,1))

with open('../data/train_df_with_sas.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open('../data/test_df_with_sas.pkl', 'rb') as f:
    test_df = pickle.load(f)

print("Train Dataframe : ")
print(train_df.head())
print("Test Dataframe : ")
print(test_df.head())

SMI_MAX_SIZE = 300
SMI_MIN_FREQ=1
with open("../data/smiles_corpus.txt", "r") as f:
    smiles_vocab = WordVocab(f, max_size=SMI_MAX_SIZE, min_freq=SMI_MIN_FREQ)

print("Built vocabulary with size: ", len(smiles_vocab))

test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties_list=config['properties'])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import QED, Descriptors, Crippen
print(config)

model = load_model(config,model_file_name='best_model.pt')

properties, samples, og_smiles = sample_a_bunch(model, test_loader, greedy=False,temperature=args.temp)
train_SMILES = train_df['smiles'].tolist()
metrics = compute_metrics(train_SMILES, og_smiles, samples)
for key in metrics:
    print(f"{key}: {metrics[key]}")
metrics_path = os.path.join('../checkpoints/'+ config['run_name'], f"final_metrics_temp_{args.temp}.csv")
with open(metrics_path, "w") as f:
    f.write("Metric,Value\n")
    for key in metrics:
        f.write(f"{key},{metrics[key]}\n")

print("Generating molecules with target properties...")

target_props = {}
if 'affinity' in config['properties']:
    target_props['affinity'] = args.affinity_targets
if 'logps' in config['properties'] or 'logp' in config['properties']:
    target_props['logps'] = args.logp_targets
if 'qeds' in config['properties'] or 'qed' in config['properties']:
    target_props['qeds'] = args.qed_targets
if 'tpsas' in config['properties'] or 'tpsa' in config['properties']:
    target_props['tpsas'] = args.tpsa_targets
if 'sas' in config['properties']:
    target_props['sas'] = args.sas_targets

queries = []
property_vectors = []

prop_names = config['properties']

prop_values = [target_props.get(p, [0]) for p in prop_names]

for combo in itertools.product(*prop_values):
    key = '_'.join(str(v) for v in combo)
    t_list = []
    for i, p in enumerate(prop_names):
        val = combo[i]
        if p == 'affinity':
            t_list.append(affinity_scaler.transform([[val]]).flatten()[0])
        elif p in ['logps', 'logp']:
            t_list.append(logp_scaler.transform([[val]]).flatten()[0])
        elif p in ['qeds', 'qed']:
            t_list.append(qed_scaler.transform([[val]]).flatten()[0])
        elif p in ['tpsas', 'tpsa']:
            t_list.append(tpsas_scaler.transform([[val]]).flatten()[0])
        elif p == 'sas':
            t_list.append(sas_scaler.transform([[val]]).flatten()[0])
    
    queries.append(key)
    property_vectors.append(torch.Tensor(t_list))

property_vectors = torch.stack(property_vectors, dim=0)

sampler = Sampler(model, smiles_vocab, temperature=args.temp)
results_dict = {}

for key, v in zip(queries, property_vectors):
    start_time = time.time()
    print(f"Generating for {key}...")
    p = v.repeat(args.num_samples, 1)
    samples = sampler.sample(p, greedy=False)
    end_time = time.time()
    print(f"Generated {len(samples)} samples for {key} in {end_time - start_time:.2f} seconds.")
    results_dict[key] = samples

checkpoint_dir = '../checkpoints/' + config['run_name']
os.makedirs(checkpoint_dir, exist_ok=True)

with open(os.path.join(checkpoint_dir, f'generated_molecules_temp_{args.temp}.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)

print(f"Generated molecules saved to: {checkpoint_dir}/generated_molecules_temp_{args.temp}.pkl")