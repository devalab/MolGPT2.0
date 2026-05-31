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
print("Loaded libraries.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties to use (e.g., --properties affinity logps)')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints and logs')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units in feedforward layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
args = parser.parse_args()
print("Properties to use: ", args.properties)

config = {
    'batch_size' : args.batch_size,
    'd_model': args.d_model,
    'n_heads': args.n_heads,
    'n_layers': args.n_layers,
    'hidden_units': args.hidden_units,
    'lr': args.lr,
    'epochs': args.epochs,
    'properties': sorted(args.properties)
}
if args.checkpoint_dir is not None:
    config['run_name'] = args.checkpoint_dir
else:
    config['run_name'] = "encoder_decoder"+ "_".join(prop for prop in config['properties'])



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

# PyTorch Dataset for our model
class CustomTargetDataset(Dataset):
    def __init__(self, df, vocab, properties_list):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.props = properties_list    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi = row['smiles']
        seq = self.vocab.to_seq(split(smi), seq_len=SMI_MAX_SIZE, with_eos=True, with_sos=True)
        prop_vals = [row[p] for p in self.props]
        return {
            "smiles_rep": torch.tensor(seq, dtype=torch.long),
            "properties": torch.tensor(prop_vals, dtype=torch.float32),
            "smiles": smi
        }

# Positional Encoding Layer for Transformer
class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout,n=10000):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (n ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings[:,:x.shape[1],:]
        x = self.dropout(x)
        return x

# Property Encoder for generating property embeddings
class PropertyEncoder(nn.Module):
    def __init__(self, d_model, n_properties):
        super(PropertyEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_properties)])
        self.layer_final = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_properties)])
    def forward(self, x):
        outs = [self.layer_final[i](F.relu(self.layers[i](x[:,i].unsqueeze(1)))) for i, layer in enumerate(self.layers)]
        # for i, layer in enumerate(self.layers):
        #     out = self.layers[i](x[:,i])
        #     out = F.relu(out)
        #     x = self.layer_final[i](out)        
        return torch.stack(outs, dim=1)

# Causal mask for masked attention
def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

# MolGPT2 model architecture (encoder-decoder)
class MolGPT2(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab, n_properties, hidden_units=1024, dropout=0.1):
        super(MolGPT2, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab = vocab
        self.dropout = dropout
        
        self.embed = nn.Embedding(len(vocab), d_model)
        self.smile_pe = PositionalEncodings(SMI_MAX_SIZE, d_model, dropout)
        
        self.trfmLayer = nn.TransformerDecoderLayer(d_model=d_model,
                                                    nhead=n_heads,
                                                    dim_feedforward=hidden_units,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    activation="gelu")
        self.trfm = nn.TransformerDecoder(decoder_layer=self.trfmLayer,
                                          num_layers=n_layers,
                                          norm=nn.LayerNorm(d_model))
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(vocab))

        # Property side: per-property projection + encoder
        self.property_encoder = PropertyEncoder(d_model, n_properties=n_properties)
        self.prop_pe = PositionalEncodings(n_properties, d_model, dropout)
        self.prop_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_units,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.prop_encoder = nn.TransformerEncoder(
            encoder_layer=self.prop_enc_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        
    def forward(self, x, property):
        # Encode properties -> contextual memory
        prop_feats = self.property_encoder(property)          # (B, P, D)
        prop_feats = self.prop_pe(prop_feats)                 # (B, P, D)
        memory = self.prop_encoder(prop_feats)                # (B, P, D)
        
        x = self.embed(x)
        x = self.smile_pe(x)
    
        mask = set_up_causal_mask(x.shape[1]).to(x.device)
        x = self.trfm(tgt=x,
                      memory=memory,
                      tgt_mask=mask,
                      )
        x = self.ln_f(x)
        x = self.classifier(x)
        return x

# Training and validation steps
def train_step(model, data_loader, optimizer,epoch):
    running_loss = []
    model.to(device)
    model.train()
    for i, data in enumerate((data_loader)):
        # data = {k: v.to(device) for k, v in data.items()}
        data['smiles_rep'] = data['smiles_rep'].to(device)
        data['properties'] = data['properties'].to(device)
        optimizer.zero_grad()
        out = model(data['smiles_rep'], data['properties'])
        out = out[:,:-1,:]
        y = data['smiles_rep'][:,1:]
        loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),y.contiguous().view(-1))
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        running_loss.append(loss.item())
        if (i+1) % 10 == 0:
            print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), loss.item() ), end='\r',flush=True)
    return np.mean(running_loss)
        
def val_step(model, data_loader, epoch):
    running_loss = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate((data_loader)):
            # data = {k: v.to(device) for k, v in data.items()}
            data['smiles_rep'] = data['smiles_rep'].to(device)
            data['properties'] = data['properties'].to(device)
            
            out = model(data['smiles_rep'], data['properties'])
            out = out[:,:-1,:]
            y = data['smiles_rep'][:,1:]
            loss = F.cross_entropy(out.contiguous().view(-1, len(smiles_vocab)),y.contiguous().view(-1))
            running_loss.append(loss.item())
            if (i+1) % 10 == 0:
                print( 'Validating Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(data_loader), loss.item() ), end='\r',flush=True)
    return np.mean(running_loss)


# Function to save model and config
def save_model(model, config,model_file_name='model.pt'):
    path_dir = '../checkpoints/'+ config['run_name']
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    model_path = path_dir + '/' + model_file_name
    config_path = path_dir + '/config.yaml'
    torch.save(model.state_dict(), model_path)
    with open(config_path,'w') as yaml_file:
        yaml.dump(dict(config), yaml_file)
        

# Class for sampling molecules
class Sampler:
    def __init__(self, model, vocab, temperature=1.0):
        self.model = model
        self.vocab = vocab
        self.temperature = temperature
    
    def sample(self, properties, greedy=False):
        samples = []
        with torch.no_grad():
            property = properties.to(device)
            sos_id = self.vocab.stoi.get("<sos>", 0)
            smiles_seq = torch.full((property.shape[0], 1), sos_id).long().to(device)
            # print(smiles_seq)
            # return
            
            for i in range(SMI_MAX_SIZE):
                logits = self.model.forward(smiles_seq, property) / self.temperature
                # print(logits.shape)
                probs = F.softmax(logits[:,-1], dim= -1)
                # print(probs.shape)
                # break
                if greedy:
                    pred_id = torch.argmax(probs, dim= -1)
                    pred_id = pred_id.unsqueeze(1)
                else:
                    pred_id = torch.multinomial(probs, num_samples=1)
                # print(pred_id.shape)
                # break
                smiles_seq = torch.cat([smiles_seq, pred_id], dim=1)
                
            for i in range(len(smiles_seq)):
                smile = self.vocab.from_seq(smiles_seq[i].cpu().numpy())
                final_smile = ""
                # skip first start token since we always initialize with it
                start_idx = 1
                for char in smile[start_idx:]:
                    if char == "<eos>" or char == "<end>" or char == "<pad>":
                        break
                    final_smile += char
                samples.append(final_smile)
        return samples
            
def sample_a_bunch(model, dataloader, greedy=False, temperature=1.0):
    sampler = Sampler(model, smiles_vocab, temperature=temperature)
    model.eval()
    samples = []
    properties = []
    og_smiles = []
    with torch.no_grad():
        for i, data in enumerate((dataloader)):
            # data = {k: v.to(device) for k, v in data.items()}
            smiles = sampler.sample(data['properties'].to(device), greedy=greedy)
            properties += data['properties'].cpu().numpy().tolist()
            ogs = data['smiles']
            samples += smiles
            og_smiles += ogs
            print( 'Sampling: iteration: {}/{}'.format(i, len(dataloader)), end='\r',flush=True)
            if len(samples) >= 1000:
                break
    return np.array(properties), samples, og_smiles

# Code for computing metrics like validity, novelty, uniqueness, and internal diversity
def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def compute_metrics(train_SMILES, test_SMILES, predicted_SMILES):
    # Compute validity
    valid_predicted = [smiles for smiles in predicted_SMILES if is_valid_smiles(smiles)]
    validity = len(valid_predicted) / len(predicted_SMILES) if predicted_SMILES else 0

    # Compute novelty
    novel_predicted = [smiles for smiles in valid_predicted if smiles not in train_SMILES]
    novelty = len(novel_predicted) / len(valid_predicted) if valid_predicted else 0

    # Compute uniqueness
    unique_predicted = set(valid_predicted)
    uniqueness = len(unique_predicted) / len(valid_predicted) if valid_predicted else 0

    # Internal Diversity
    mols = [Chem.MolFromSmiles(smi) for smi in valid_predicted]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]
    internal_diversity = 0.0
    if len(fps) > 1:
        sims = []
        for i in range(len(fps)-1):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        if len(sims) > 0:
            internal_diversity = 1.0 - np.mean(sims)

    return {
        'Validity': validity,
        'Novelty': novelty,
        'Uniqueness': uniqueness,
        'Internal Diversity': internal_diversity
    }

def load_model(config,model_file_name='model.pt'):
    path_dir = '../checkpoints/'+ config['run_name']
    model_path = path_dir + '/' + model_file_name
    model = MolGPT2(d_model=config['d_model'], 
                   n_heads=config['n_heads'], 
                   n_layers=config['n_layers'], 
                   vocab=smiles_vocab, 
                   n_properties=len(config['properties']), 
                   hidden_units=config['hidden_units'],
                   dropout=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)
    num_gpus = torch.cuda.device_count()
    print("No of GPUs available", num_gpus)
    try:
        model = torch.nn.parallel.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model = MolGPT2(d_model=config['d_model'], 
                   n_heads=config['n_heads'], 
                   n_layers=config['n_layers'], 
                   vocab=smiles_vocab, 
                   n_properties=len(config['properties']), 
                   hidden_units=config['hidden_units'],
                   dropout=0.1)
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.parallel.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    model.eval()
    return model

def run(config):
    PROPERTIES = config['properties']
    train_dataset = CustomTargetDataset(train_df, smiles_vocab, properties_list=PROPERTIES)
    test_dataset = CustomTargetDataset(test_df, smiles_vocab, properties_list=PROPERTIES)
    train_SMILES = train_df['smiles'].tolist()

    batch_size = config['batch_size'] # Define your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    data = next(iter(train_loader))
 
    model = MolGPT2(d_model=config['d_model'], 
                   n_heads=config['n_heads'], 
                   n_layers=config['n_layers'], 
                   vocab=smiles_vocab, 
                   n_properties=len(PROPERTIES), 
                   hidden_units=config['hidden_units'],
                   dropout=0.1)
    model = torch.nn.parallel.DataParallel(model)

    os.makedirs('../checkpoints', exist_ok=True)
    path_dir = '../checkpoints/'+ config['run_name']
    os.makedirs(path_dir, exist_ok=True)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(os.path.join(path_dir, "model_params.txt"), "w") as f:
        f.write(f"Total trainable parameters: {total_params}\n")
        
    start_epoch = 0
    num_epochs_file = os.path.join(path_dir, "num_epochs.txt")
    if os.path.exists(num_epochs_file):
        try:
            with open(num_epochs_file, "r") as f:
                content = f.read().strip()
                if content.isdigit():
                    start_epoch = int(content)
        except Exception:
            pass

    tl = []
    vl = []
    metric_records = []
    epoch_records = []

    #load previously trained model if exists and resume training from there
    last_model_path = os.path.join(path_dir, 'last_model.pt')
    if os.path.exists(last_model_path):
        print(f"Loading last model from {last_model_path} to resume from epoch {start_epoch}")
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        
        losses_path = os.path.join(path_dir, "losses.pkl")
        if os.path.exists(losses_path):
            with open(losses_path, "rb") as f:
                saved_losses = pickle.load(f)
                tl = saved_losses.get("train_losses", [])
                vl = saved_losses.get("val_losses", [])
                
        metrics_path = os.path.join(path_dir, "metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                saved_metrics = pickle.load(f)
                metric_records = saved_metrics.get("metrics", [])
                epoch_records = saved_metrics.get("epochs", [])
    elif os.path.exists(os.path.join(path_dir, 'best_model.pt')):
        print(f"Loading model from {os.path.join(path_dir, 'best_model.pt')}")
        model.load_state_dict(torch.load(os.path.join(path_dir, 'best_model.pt'), map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)
    num_gpus = torch.cuda.device_count()
    print("No of GPUs available", num_gpus)

    wandb.init(project="molgpt2.0 FINAL", config=config, name=config['run_name'])
    wandb.watch(models=model, log_freq=100)
    print(config)

    sampler = Sampler(model, smiles_vocab, temperature=1.0)
    All_samples = []
    
    best_val_loss = float('inf')
    if vl:
        best_val_loss = min(vl)
    
    for i in (range(start_epoch, config['epochs'])):
        time_start = time.time()
        train_loss = train_step(model, train_loader, optimizer,i)
        val_loss = val_step(model, test_loader, i)
        tl.append(train_loss)
        vl.append(val_loss)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)
        
        # Save last, best
        # torch.save(model.state_dict(), os.path.join(path_dir, 'last_model.pt'))
        save_model(model, config, model_file_name='last_model.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join(path_dir, 'best_val_loss.txt'), 'w') as f:
                f.write(str(best_val_loss))
            # torch.save(model.state_dict(), os.path.join(path_dir, 'best_model.pt'))
            save_model(model, config, model_file_name='best_model.pt')
            with open(os.path.join(path_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(i+1))
            
        try:
            with open(os.path.join(path_dir, "num_epochs.txt"), "w") as f:
                f.write(str(i+1))
        except Exception:
            pass
            
        with open(os.path.join(path_dir, "losses.pkl"), "wb") as f:
            pickle.dump({"train_losses": tl, "val_losses": vl}, f)
        
        #for constantly plotting loss function    
        plt.figure()
        plt.plot(tl, label='Train Loss')
        plt.plot(vl, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(path_dir, "loss_plot.png"))
        plt.close()
        
        if (i+1) % 10 == 0:
            properties, pred_SMILES, test_SMILES  = sample_a_bunch(model, test_loader, greedy=False, temperature=1.0)
            results = compute_metrics(train_SMILES, test_SMILES, pred_SMILES)
            for key in results:
                print(f"{key}: {results[key]}")
            
            epoch_records.append(i+1)
            metric_records.append(results)
            with open(os.path.join(path_dir, "metrics.pkl"), "wb") as f:
                pickle.dump({"epochs": epoch_records, "metrics": metric_records}, f)
                
            plt.figure(figsize=(10, 6))
            for key in results.keys():
                vals = [m[key] for m in metric_records]
                plt.plot(epoch_records, vals, label=key)
            plt.xlabel('Epochs')
            plt.ylabel('Metrics')
            plt.legend()
            plt.savefig(os.path.join(path_dir, "metrics_plot.png"))
            plt.close()
            
            df = pd.DataFrame({"SMILES":pred_SMILES})
            df.to_csv(os.path.join(path_dir, f'sampled_mols_epoch.txt'), index=False)
            
        time_end = time.time()
        print(f"Time Taken for Epoch {i} : {time_end - time_start} seconds",flush=True)
            
    # After training
    with open(os.path.join(path_dir, "losses.pkl"), "wb") as f:
        pickle.dump({"train_losses": tl, "val_losses": vl}, f)
        
    plt.figure()
    plt.plot(tl, label='Train Loss')
    plt.plot(vl, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path_dir, "loss_plot.png"))
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    run(config)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")