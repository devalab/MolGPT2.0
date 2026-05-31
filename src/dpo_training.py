import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import pickle
import sys
import argparse

from tqdm.auto import tqdm

from build_corpus import build_corpus
from build_vocab import WordVocab
from utils import split
from train_models import CustomTargetDataset, PositionalEncodings, PropertyEncoder,set_up_causal_mask, MolGPT2, save_model, load_model, Sampler, sample_a_bunch, compute_metrics

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import RDConfig

import sklearn
from sklearn.preprocessing import MinMaxScaler

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Loaded libraries.")

df = pd.read_csv('../data/lck_dockstring_data1.csv')

parser = argparse.ArgumentParser(description='Train dpo model')
parser.add_argument('--properties', nargs='+', required=True, help='Properties used to train the base model (e.g., --properties affinity logps qeds sas tpsas)')
parser.add_argument('--checkpoint_dir', type=str, default=None,help='Directory to store DPO outputs (models, plots, results)')
parser.add_argument('--preference_properties', nargs='+', default =['affinity'],help='Properties to use for preference dataset (e.g., --preference_properties affinity)')
parser.add_argument('--epochs', type=int, default=10, help='Number of DPO training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units in feedforward layers')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
parser.add_argument('--ipo', action='store_true', help='Whether to use IPO loss instead of DPO loss')
parser.add_argument('--beta', type=float, default=0.11, help='Beta parameter for DPO loss')
parser.add_argument('--base_model_dir', type=str, default=None, help='Directory to load base model from (if not specified, will look for model matching properties in `checkpoints` directory)')
args = parser.parse_args()
print("Model properties: ", args.properties)
print("Preference properties: ", args.preference_properties)
print("Base model dir: ", args.base_model_dir)
print("Checkpoint dir: ", args.checkpoint_dir)

config = {
    'batch_size' : args.batch_size,
    'd_model': args.d_model,
    'n_heads': args.n_heads,
    'n_layers': args.n_layers,
    'hidden_units': args.hidden_units,
    'lr': args.lr,
    'epochs': args.epochs,
    'properties': sorted(args.properties),  # Properties used by the base model
    'preference_properties': sorted(args.preference_properties),  # Properties for preference data
    'ipo':False,
    'beta': args.beta
}
if args.base_model_dir:
    config['base_model_dir'] = args.base_model_dir
else:
    config['base_model_dir'] = "encoder_decoder_" + "_".join(prop for prop in config['properties'])
config['run_name'] = config['base_model_dir'] # to load base model

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

SMI_MAX_SIZE = 300
SMI_MIN_FREQ=1
with open("../data/smiles_corpus.txt", "r") as f:
    smiles_vocab = WordVocab(f, max_size=SMI_MAX_SIZE, min_freq=SMI_MIN_FREQ)

class PreferenceDataset(Dataset):
    def __init__(self, preference_data, vocab, property_names):
        self.preference_Data = preference_data
        self.smiles_vocab = vocab
        self.property_names = property_names
        self.build()
          
    
    def build(self):
        smiles_good, smiles_bad, properties = [],[],[]
        
        for i, row in enumerate(self.preference_Data):
            smi = row[0]
            target_props = row[1]
            good_smi = row[2][0]
            good_prop = row[2][2]
            bad_smi = row[3][0]
            bad_prop = row[3][2]
            smiles_good.append(self.smiles_vocab.to_seq(split(good_smi), seq_len=SMI_MAX_SIZE, with_eos=True, with_sos=True))
            smiles_bad.append(self.smiles_vocab.to_seq(split(bad_smi), seq_len=SMI_MAX_SIZE, with_eos=True, with_sos=True))
            properties.append(target_props)
            
        self.smiles_good_encodings = torch.tensor(smiles_good)
        self.smiles_bad_encodings = torch.tensor(smiles_bad)
        self.properties = torch.tensor(properties, dtype=torch.float32)
        print("dataset built")
        
    def __len__(self):
        return len(self.properties)
    
    def __getitem__(self, index):
        return {
            "smiles_good": self.smiles_good_encodings[index],
            "smiles_bad": self.smiles_bad_encodings[index],
            "properties": self.properties[index]
        }

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta,
                    label_smoothing: float = 0.1,
                    ipo: bool = False
                    ):
  
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        
    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    # losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def calculate_logprobs(model, smiles_rep, properties):
    
    out = model(smiles_rep, properties)
    ids = smiles_rep[:,1:]
    out = out[:,:-1,:]
    logitprobs = F.log_softmax(out, dim= -1)
    finalprobs =  logitprobs[np.arange(logitprobs.shape[0])[:,None], np.arange(logitprobs.shape[1])[None,:], ids]
    logprobs = finalprobs.sum(dim=-1)
    return logprobs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def train_step(model, ref_model, data_loader, optimizer,epoch):
    #print("BETA", BETA)
    running_loss = []
    model.to(device)
    model.train()
    for i, data in enumerate(data_loader):
        data = {k: v.to(device) for k, v in data.items()}
        
        optimizer.zero_grad()    
            
        policy_chosen_logps = calculate_logprobs(model, data['smiles_good'], data['properties'])
        policy_rejected_logps = calculate_logprobs(model, data['smiles_bad'], data['properties'])

        reference_chosen_logps = calculate_logprobs(ref_model, data['smiles_good'], data['properties'])
        reference_rejected_logps = calculate_logprobs(ref_model, data['smiles_bad'], data['properties'])

        losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=config['beta'], ipo=config['ipo'])
        losses = losses.mean(dim=-1)
        losses.backward()
        optimizer.step()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        running_loss.append(losses.item())
        print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), losses.item() ), end='\r')
        
    return np.mean(running_loss)

def val_step(model, ref_model, data_loader,epoch):
    running_loss = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            
            policy_chosen_logps = calculate_logprobs(model, data['smiles_good'], data['properties'])
            policy_rejected_logps = calculate_logprobs(model, data['smiles_bad'], data['properties'])

            reference_chosen_logps = calculate_logprobs(ref_model, data['smiles_good'], data['properties'])
            reference_rejected_logps = calculate_logprobs(ref_model, data['smiles_bad'], data['properties'])

            losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=config['beta'], ipo=config['ipo'])
            losses = losses.mean(dim=-1)
            
            running_loss.append(losses.item())
            print( 'Validating Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), losses.item() ), end='\r')
        
    return np.mean(running_loss)

def run(config,preference_dataset):
    PROPERTIES = config['properties'] 
    
    batch_size = config['batch_size']

    train_SMILES = train_df['smiles'].tolist()

    train_size = int(0.8 * len(preference_dataset))
    test_size = len(preference_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(preference_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    
    model = load_model(config, model_file_name='best_model.pt')
    ref_model = load_model(config, model_file_name='best_model.pt')
    ref_model = freeze_model(ref_model)
    model.to(device)
    ref_model.to(device)

    lr = config['lr']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    tl = []
    vl = []
    
    model_props_str = "_".join(config['properties'])
    pref_props_str = "_".join(config['preference_properties'])
    pref_tag = f"_pref_{pref_props_str}"

    if args.checkpoint_dir:
        config['run_name'] = args.checkpoint_dir
    elif config['ipo']:
        config['run_name'] = f"IPO_encoder_decoder_{model_props_str}{pref_tag}"
    else:
        config['run_name'] = f"DPO_encoder_decoder_{model_props_str}{pref_tag}"

    os.makedirs("../checkpoints", exist_ok=True)
    os.makedirs(os.path.join("../checkpoints", config['run_name']), exist_ok=True)
    
    wandb.init(project="molgpt2.0 FINAL", config=config, name=config['run_name'])
    wandb.watch(models=model, log_freq=100)
    print(config)

    sampler = Sampler(model, smiles_vocab)
    All_samples = []
    og_test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties_list=config['properties'])
    og_test_loader = DataLoader(og_test_dataset, batch_size=1024, shuffle=True, num_workers=12)
    
    best_val_loss = float('inf')

    for i in range(config['epochs']):
        
        properties, pred_SMILES, test_SMILES = sample_a_bunch(model, og_test_loader, greedy=False, temperature=1.0)
        results = compute_metrics(train_SMILES, test_SMILES, pred_SMILES)
        print(results)
        for key in results:
            wandb.log({key: results[key]}, step=i)
                
        train_loss = train_step(model, ref_model, train_loader, optimizer, i)
        val_loss = val_step(model, ref_model, test_loader, i)
        
        tl.append(train_loss)
        vl.append(val_loss)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)
        
        save_model(model, config)
        checkpoint_dir = os.path.join("../checkpoints", config['run_name'])
        
        save_model(model, config,model_file_name=f"last_model.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, model_file_name='best_model.pt')
    
    with open(os.path.join(checkpoint_dir, "losses.pkl"), "wb") as f:
        pickle.dump({"train_losses": tl, "val_losses": vl}, f)
        
    plt.figure()
    plt.plot(tl, label='Train Loss')
    plt.plot(vl, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "loss_plot.png"))
    plt.close()
        
if __name__ == "__main__":
    preference_file = "../data/PreferenceData_" + "_".join(config['preference_properties']) + ".pkl"
    with open(preference_file, 'rb') as f:
        preference_data = pickle.load(f)
    preference_dataset = PreferenceDataset(preference_data, smiles_vocab, property_names=config['preference_properties'])

    run(config, preference_dataset)