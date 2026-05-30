
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import sys
import pickle
import argparse

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import QED, Descriptors, Crippen

import dockstring
from dockstring import load_target
import multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler

import math

import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.title_fontsize': 15,
    'legend.fontsize': 15,
    'figure.autolayout': True
})



sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


parser = argparse.ArgumentParser(description='Analyze generated molecules')
parser.add_argument('--checkpoint_dir', type=str, required=True, 
                    help='Directory containing generated_results.pkl')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties used (e.g., --properties affinity logps qeds sas tpsas)')
parser.add_argument('--plot_targets', nargs='*', default=[],
                    help='Specific target values to plot for properties (e.g., --plot_targets affinity=9.0,6.0 logps=2.0)')
parser.add_argument('--plot_kde_separately', action='store_true',
                    help='Plot KDEs separately for each target value')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature at which molecules were sampled')
args = parser.parse_args()


# Load the original dataset to fit scalers
df = pd.read_csv('../data/lck_dockstring_data1.csv')

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


def calc_properties(properties, smiles):
    """Calculate molecular properties for a list of SMILES."""
    qeds = []
    logps = []
    tpsas = []
    sas = []
    molwt = []
    props = []
    valid_smi = []
    
    for prop, smi in zip(properties, smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            if mol is not None:
                qed = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                sa = sascorer.calculateScore(mol)
                mw = Descriptors.MolWt(mol)
                
                qeds.append(qed)
                logps.append(logp)
                tpsas.append(tpsa)
                sas.append(sa)
                molwt.append(mw)
                props.append(prop)    
                valid_smi.append(smi)            
        except:
            pass
                
    return qeds, logps, tpsas, sas, molwt, np.array(props), valid_smi


def dock_smile(smile):
    """Calculate binding affinity for a single SMILES using dockstring."""
    target = load_target('LCK')
    try:
        score, _ = target.dock(smile)
    except:
        score = None
    return score

def calculate_binding_affinities(smiles_list, batch_size=16):
    """Calculate binding affinities for a list of SMILES using multiprocessing."""
    print(f"Calculating binding affinities for {len(smiles_list)} molecules...")
    
    # Pad to multiple of batch_size
    smiles_array = np.array(smiles_list)
    if len(smiles_array) % batch_size != 0:
        padding = batch_size - (len(smiles_array) % batch_size)
        smiles_array = np.concatenate([smiles_array, smiles_array[:padding]])
    
    smiles_batches = smiles_array.reshape(batch_size, -1)
    
    all_scores = []
    for batch in tqdm(smiles_batches, desc="Docking batches"):
        with mp.Pool(mp.cpu_count()) as pool:
            scores = pool.map(dock_smile, batch)
        all_scores.extend(scores)
    
    # Return only the original length (remove padding)
    all_scores = all_scores[:len(smiles_list)]
    return np.array(all_scores)


def load_and_calculate_properties(checkpoint_dir, prop_names, temp):
    """Load generated molecules and calculate their properties.
    
    Only compute docking affinities if 'affinity' is present in prop_names.
    """
     
     # Load generated results
    results_path = os.path.join(checkpoint_dir, f'generated_molecules_temp_{temp}.pkl')
    with open(results_path, 'rb') as f:
         results_dict = pickle.load(f)
     
    print(f"Loaded {len(results_dict)} query results")
     
    # Calculate properties for each query
    final_results = {}
   
    compute_affinities = ('affinity' in prop_names)
    
    train_smiles = set()
    try:
        df_train = pd.read_csv('../data/lck_dockstring_data1.csv')
        if 'smiles' in df_train.columns:
            train_smiles = set(df_train['smiles'].tolist())
    except Exception as e:
        print(f"Warning: Could not load training SMILES for novelty calculation: {e}")

    generation_metrics = []
    total_gen = 0
    total_val = 0
    all_valid_smi_set = set()
     
    for key, samples in tqdm(results_dict.items(), desc="Calculating properties"):
        print(f"\nProcessing {key}...")
         
        # Calculate chemical properties
        qeds, logps, tpsas, sas, molwt, og_props, smi = calc_properties([-1]*len(samples), samples)
         
        # Calculate binding affinities using dockstring only if requested
        if compute_affinities:
            affinities = calculate_binding_affinities(smi)
        else:
          affinities = [None] * len(smi)
         
        final_results[key] = [logps, qeds, sas, tpsas, affinities, samples]
        
        # Calculate generation metrics
        num_gen = len(samples)
        num_val = len(smi)
        validity = num_val / num_gen if num_gen > 0 else 0
        uniq_set = set(smi)
        num_uniq = len(uniq_set)
        uniqueness = num_uniq / num_val if num_val > 0 else 0
        novel_set = uniq_set - train_smiles
        num_novel = len(novel_set)
        novelty = num_novel / num_uniq if num_uniq > 0 else 0
        
        generation_metrics.append({
            'combo': str(key),
            'num_generated': num_gen,
            'num_valid': num_val,
            'validity': validity,
            'num_unique': num_uniq,
            'uniqueness': uniqueness,
            'num_novel': num_novel,
            'novelty': novelty
        })
        total_gen += num_gen
        total_val += num_val
        all_valid_smi_set.update(smi)
     
    # Save final results with all properties
    output_path = os.path.join(checkpoint_dir, f'generated_results_with_properties_temp_{temp}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(final_results, f)
     
    print(f"\nSaved results to: {output_path}")

    # Compute ALL metrics
    total_uniq_set = all_valid_smi_set
    total_uniq = len(total_uniq_set)
    total_novel_set = total_uniq_set - train_smiles
    total_novel = len(total_novel_set)
    generation_metrics.append({
        'combo': 'ALL',
        'num_generated': total_gen,
        'num_valid': total_val,
        'validity': total_val / total_gen if total_gen > 0 else 0,
        'num_unique': total_uniq,
        'uniqueness': total_uniq / total_val if total_val > 0 else 0,
        'num_novel': total_novel,
        'novelty': total_novel / total_uniq if total_uniq > 0 else 0
    })
    
    gen_df = pd.DataFrame(generation_metrics)
    gen_metrics_path = os.path.join(checkpoint_dir, f'generation_metrics_temp_{temp}.csv')
    gen_df.to_csv(gen_metrics_path, index=False)
    print(f"Saved generation metrics to: {gen_metrics_path}")
     
    return final_results


def create_dataframe(results_dict, prop_names):
    """Create a DataFrame from results dictionary."""
    data = []
    columns = []
    
    for prop in prop_names:
        columns.append(f'Target {prop}')
    for prop in ['logps', 'qeds', 'sas', 'tpsas', 'affinity']:
        columns.append(f'Predicted {prop}')
    columns.append('key')
    
    for key in results_dict:
        # keys were created as strings (e.g. '3.5_0.7_...').
        # Try to convert each key-part to float so Target columns are numeric.
        raw_keys = key.split('_')
        conv_keys = []
        for k in raw_keys:
            try:
                conv_keys.append(float(k))
            except Exception:
                conv_keys.append(k)

        for i in range(len(results_dict[key][0])):
            # Filter out None affinities
            if (results_dict[key][4][i] is not None) or ('affinity' not in prop_names):
                row = conv_keys + [
                    results_dict[key][0][i],  # logps
                    results_dict[key][1][i],  # qeds
                    results_dict[key][2][i],  # sas
                    results_dict[key][3][i],  # tpsas
                    results_dict[key][4][i],  # affinity
                    key
                ]
                data.append(row)
    
    data_df = pd.DataFrame(data, columns=columns)

    # Ensure numeric columns are numeric (coerce non-convertible values to NaN).
    for col in columns:
        if col.startswith('Target') or col.startswith('Predicted'):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    return data_df

def _gaussian_pdf(x, mu, sigma):
    coef = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * np.exp(exponent)

def compute_metrics_for_property(data_df, prop_name, checkpoint_dir, temp, sigma=0.05):
    """Compute MAE, variance, and KL divergence (predicted vs Gaussian(target,sigma))
    for each unique target property value.

    Groups rows by unique target values and computes metrics within each group.
    Saves results to metrics_<prop_name>.csv in checkpoint_dir and returns the DataFrame.
    """
    target_col = f'Target {prop_name}'
    pred_col = f'Predicted {prop_name}'

    if target_col not in data_df.columns or pred_col not in data_df.columns:
        print(f"Skipping {prop_name}: columns not found")
        return None

    df = data_df[[target_col, pred_col]].dropna()
    if df.shape[0] == 0:
        print(f"No data for {prop_name}")
        return None

    # Get unique target values and sort them
    unique_targets = sorted(df[target_col].unique())
    results = []
    eps = 1e-12

    for target_val in unique_targets:
        # Get all rows with this target value
        mask = df[target_col] == target_val
        tvals = df.loc[mask, target_col].values
        preds = df.loc[mask, pred_col].values

        if len(preds) == 0:
            results.append({
                'target_value': target_val,
                'count': 0,
                'mae': np.nan,
                'var_pred': np.nan,
                'std_pred': np.nan,
                'mean_pred': np.nan,
                'kl_div': np.nan,
                'success_rate_05': np.nan,
                'success_rate_10': np.nan
            })
            continue

        # MAE between predicted and target
        mae = np.mean(np.abs(preds - tvals))
        
        # Variance of predictions
        var_pred = np.var(preds)

        # Standard deviation of predictions
        std_pred = np.std(preds)

        # Mean of predictions
        mean_pred = np.mean(preds)

        # Success rates
        sr_05 = np.mean(np.abs(preds - tvals) <= 0.5)
        sr_10 = np.mean(np.abs(preds - tvals) <= 1.0)

        # KL divergence between empirical predicted distribution and Gaussian(target_val, sigma)
        # empirical p from histogram
        counts, edges = np.histogram(preds, bins=20)
        p_counts = counts.astype(float)
        if p_counts.sum() == 0:
            kl = np.nan
        else:
            p_prob = p_counts / p_counts.sum()
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Gaussian centered at the target value with std=sigma
            q_vals = _gaussian_pdf(centers, target_val, sigma)
            q_prob = q_vals / (q_vals.sum() + eps)
            p_safe = p_prob + eps
            q_safe = q_prob + eps
            kl = np.sum(p_safe * np.log(p_safe / q_safe))

        results.append({
            'target_value': target_val,
            'count': int(len(preds)),
            'mae': float(mae),
            'var_pred': float(var_pred),
            'std_pred': float(std_pred),
            'mean_pred': float(mean_pred),
            'kl_div': float(kl),
            'success_rate_05': float(sr_05),
            'success_rate_10': float(sr_10)
        })

    res_df = pd.DataFrame(results)
    out_path = os.path.join(checkpoint_dir, f'metrics_{prop_name}_temp_{temp}.csv')
    res_df.to_csv(out_path, index=False)
    print(f"Saved metrics for {prop_name} to: {out_path}")
    return res_df

def compute_all_metrics(data_df, prop_names, checkpoint_dir, temp):
    # compute metrics for each property available in the dataframe
    for prop in ['logps','qeds','sas','tpsas','affinity']:
        compute_metrics_for_property(data_df, prop, checkpoint_dir, temp)

def generate_molecule_images(checkpoint_dir, temp):
    from rdkit.Chem import Draw
    img_dir = os.path.join(checkpoint_dir, f'molecule_images_temp_{temp}')
    os.makedirs(img_dir, exist_ok=True)
    
    pkl_path = os.path.join(checkpoint_dir, f'generated_results_with_properties_temp_{temp}.pkl')
    if not os.path.exists(pkl_path):
        print(f"Skipping molecule images, {pkl_path} not found.")
        return
        
    with open(pkl_path, 'rb') as f:
        results_dict = pickle.load(f)
        
    print("\nGenerating sample molecule images...")
    for key, val in results_dict.items():
        smiles_list = val[-1] 
        mols = []
        for smi in smiles_list:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                mols.append(m)
            if len(mols) == 2:
                break
                
        if mols:
            safe_key = str(key).replace('/', '_')
            img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(300, 300), returnPNG=False)
            img.save(os.path.join(img_dir, f'{safe_key}.png'))
            
    print(f"Saved sample molecule images to {img_dir}")


def plot_single_property_kdes(data_df, prop_names, target_props, results_dir, temp, plot_targets_dict={}, plot_kde_separately=False):
    """Generate single property KDE plots."""
    print("\nGenerating single property KDE plots...")
    
    if 'logps' in target_props or 'logp' in target_props:
        prop_key = 'logps' if 'logps' in prop_names else 'logp'
        target_col = f"Target {prop_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if prop_key in plot_targets_dict:
            plt_df = plt_df[plt_df[target_col].isin(plot_targets_dict[prop_key])]
            target_suffix = "_" + "_".join(map(str, plot_targets_dict[prop_key]))
        
        if plt_df.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            if plot_kde_separately:
                unique_targets = sorted(plt_df[target_col].unique())
                #colors = sns.color_palette("husl", len(unique_targets))
                for idx, target_val in enumerate(unique_targets):
                    subset = plt_df[plt_df[target_col] == target_val]["Predicted logps"].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(
                            subset,
                            fill=True, alpha=0.5, linewidth=1,
                             bw_adjust=2, label=str(target_val)
                        )
                plt.legend(title=target_col)
                plt.xlabel("Predicted logps")
            else:
                ax = sns.kdeplot(data=plt_df, x="Predicted logps", hue=target_col, 
                                fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
            # Add vertical dotted lines at target values
            for target_val in sorted(plt_df[target_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('LogP Distribution (Generated vs Target)')
            caption = "Figure: LogP Distribution (Generated vs Target). Dotted lines mark the target conditions."
            #plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'logp_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: logp_kde{target_suffix}_temp_{temp}.png")
    
    if 'qeds' in target_props or 'qed' in target_props:
        prop_key = 'qeds' if 'qeds' in prop_names else 'qed'
        target_col = f"Target {prop_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if prop_key in plot_targets_dict:
            plt_df = plt_df[plt_df[target_col].isin(plot_targets_dict[prop_key])]
            target_suffix = "_" + "_".join(map(str, plot_targets_dict[prop_key]))
            
        if plt_df.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            if plot_kde_separately:
                unique_targets = sorted(plt_df[target_col].unique())
                #colors = sns.color_palette("husl", len(unique_targets))
                for idx, target_val in enumerate(unique_targets):
                    subset = plt_df[plt_df[target_col] == target_val]["Predicted qeds"].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(
                            subset,
                            fill=True, alpha=0.5, linewidth=1,
                             bw_adjust=2, label=str(target_val)
                        )
                plt.legend(title=target_col)
                plt.xlabel("Predicted qeds")
            else:
                ax = sns.kdeplot(data=plt_df, x="Predicted qeds", hue=target_col, 
                                fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
            # Add vertical dotted lines at target values
            for target_val in sorted(plt_df[target_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('QED Distribution (Generated vs Target)')
            caption = "Figure: QED Distribution (Generated vs Target). Dotted lines mark the target conditions."
            #plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'qed_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: qed_kde{target_suffix}_temp_{temp}.png")
    
    if 'tpsas' in target_props or 'tpsa' in target_props:
        prop_key = 'tpsas' if 'tpsas' in prop_names else 'tpsa'
        target_col = f"Target {prop_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if prop_key in plot_targets_dict:
            plt_df = plt_df[plt_df[target_col].isin(plot_targets_dict[prop_key])]
            target_suffix = "_" + "_".join(map(str, plot_targets_dict[prop_key]))
            
        if plt_df.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            if plot_kde_separately:
                unique_targets = sorted(plt_df[target_col].unique())
                #colors = sns.color_palette("husl", len(unique_targets))
                for idx, target_val in enumerate(unique_targets):
                    subset = plt_df[plt_df[target_col] == target_val]["Predicted tpsas"].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(
                            subset,
                            fill=True, alpha=0.5, linewidth=1,
                             bw_adjust=2, label=str(target_val)
                        )
                plt.legend(title=target_col)
                plt.xlabel("Predicted tpsas")
            else:
                ax = sns.kdeplot(data=plt_df, x="Predicted tpsas", hue=target_col, 
                                fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
            # Add vertical dotted lines at target values
            for target_val in sorted(plt_df[target_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('TPSA Distribution (Generated vs Target)')
            caption = "Figure: TPSA Distribution (Generated vs Target). Dotted lines mark the target conditions."
            #plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'tpsa_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: tpsa_kde{target_suffix}_temp_{temp}.png")
    
    if 'sas' in target_props:
        prop_key = 'sas'
        target_col = f"Target {prop_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if prop_key in plot_targets_dict:
            plt_df = plt_df[plt_df[target_col].isin(plot_targets_dict[prop_key])]
            target_suffix = "_" + "_".join(map(str, plot_targets_dict[prop_key]))
            
        if plt_df.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            if plot_kde_separately:
                unique_targets = sorted(plt_df[target_col].unique())
                #colors = sns.color_palette("husl", len(unique_targets))
                for idx, target_val in enumerate(unique_targets):
                    subset = plt_df[plt_df[target_col] == target_val]["Predicted sas"].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(
                            subset,
                            fill=True, alpha=0.5, linewidth=1,
                             bw_adjust=2, label=str(target_val)
                        )
                plt.legend(title=target_col)
                plt.xlabel("Predicted sas")
            else:
                ax = sns.kdeplot(data=plt_df, x="Predicted sas", hue=target_col, 
                                fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
            # Add vertical dotted lines at target values
            for target_val in sorted(plt_df[target_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('SAS Distribution (Generated vs Target)')
            caption = "Figure: SAS Distribution (Generated vs Target). Dotted lines mark the target conditions."
            #plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'sas_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: sas_kde{target_suffix}_temp_{temp}.png")
    
    if 'affinity' in target_props:
        prop_key = 'affinity'
        target_col = f"Target {prop_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if prop_key in plot_targets_dict:
            plt_df = plt_df[plt_df[target_col].isin(plot_targets_dict[prop_key])]
            target_suffix = "_" + "_".join(map(str, plot_targets_dict[prop_key]))
            
        if plt_df.shape[0] > 0:
            plt.figure(figsize=(10, 6))
            if plot_kde_separately:
                unique_targets = sorted(plt_df[target_col].unique())
                # colors = sns.color_palette("husl", len(unique_targets))
                for idx, target_val in enumerate(unique_targets):
                    subset = plt_df[plt_df[target_col] == target_val]["Predicted affinity"].dropna()
                    if len(subset) > 0:
                        sns.kdeplot(
                            subset,
                            fill=True, alpha=0.5, linewidth=1,
                             bw_adjust=2, label=str(target_val)
                        )
                plt.legend(title=target_col)
            else:
                ax = sns.kdeplot(data=plt_df, x="Predicted affinity", hue=target_col, 
                                fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
            # Add vertical dotted lines at target values
            for target_val in sorted(plt_df[target_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('Binding Affinity Distribution (Generated vs Target)')
            plt.xlabel('Binding Affinity (kcal/mol)')
            caption = "Figure: Binding Affinity Distribution (Generated vs Target). Dotted lines mark the target conditions."
            #plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'affinity_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: affinity_kde{target_suffix}_temp_{temp}.png")


def plot_dual_property_kdes(data_df, prop_names, target_props, results_dir, temp, plot_targets_dict={}):
    """Generate dual property 2D KDE plots."""
    print("\nGenerating dual property KDE plots...")
    
    # TPSA vs LogP
    if ('tpsas' in target_props or 'tpsa' in target_props) and ('logps' in target_props or 'logp' in target_props):
        tpsa_key = 'tpsas' if 'tpsas' in prop_names else 'tpsa'
        logp_key = 'logps' if 'logps' in prop_names else 'logp'
        tpsa_col = f"Target {tpsa_key}"
        logp_col = f"Target {logp_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if tpsa_key in plot_targets_dict:
            plt_df = plt_df[plt_df[tpsa_col].isin(plot_targets_dict[tpsa_key])]
            target_suffix += f"_{tpsa_key}_" + "_".join(map(str, plot_targets_dict[tpsa_key]))
        if logp_key in plot_targets_dict:
            plt_df = plt_df[plt_df[logp_col].isin(plot_targets_dict[logp_key])]
            target_suffix += f"_{logp_key}_" + "_".join(map(str, plot_targets_dict[logp_key]))
        if plt_df.shape[0] > 0:
            plt_df['TPSA-LOGP'] = plt_df[tpsa_col].astype(str) + '-' + plt_df[logp_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted tpsas", y="Predicted logps", hue="TPSA-LOGP", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[tpsa_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[logp_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('TPSA vs LogP (Dual Property Conditioning)')
            caption = "Figure: TPSA vs LogP (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'tpsa_logp_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: tpsa_logp_kde{target_suffix}_temp_{temp}.png")
    
    # TPSA vs SAS
    if ('tpsas' in target_props or 'tpsa' in target_props) and 'sas' in target_props:
        tpsa_key = 'tpsas' if 'tpsas' in prop_names else 'tpsa'
        sas_key = 'sas'
        tpsa_col = f"Target {tpsa_key}"
        sas_col = f"Target {sas_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if tpsa_key in plot_targets_dict:
            plt_df = plt_df[plt_df[tpsa_col].isin(plot_targets_dict[tpsa_key])]
            target_suffix += f"_{tpsa_key}_" + "_".join(map(str, plot_targets_dict[tpsa_key]))
        if sas_key in plot_targets_dict:
            plt_df = plt_df[plt_df[sas_col].isin(plot_targets_dict[sas_key])]
            target_suffix += f"_{sas_key}_" + "_".join(map(str, plot_targets_dict[sas_key]))
        if plt_df.shape[0] > 0:
            plt_df['TPSA-SAS'] = plt_df[tpsa_col].astype(str) + '-' + plt_df[sas_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted tpsas", y="Predicted sas", hue="TPSA-SAS", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[tpsa_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[sas_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('TPSA vs SAS (Dual Property Conditioning)')
            caption = "Figure: TPSA vs SAS (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'tpsa_sas_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: tpsa_sas_kde{target_suffix}_temp_{temp}.png")
    
    # QED vs LogP
    if ('qeds' in target_props or 'qed' in target_props) and ('logps' in target_props or 'logp' in target_props):
        qed_key = 'qeds' if 'qeds' in prop_names else 'qed'
        logp_key = 'logps' if 'logps' in prop_names else 'logp'
        qed_col = f"Target {qed_key}"
        logp_col = f"Target {logp_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if qed_key in plot_targets_dict:
            plt_df = plt_df[plt_df[qed_col].isin(plot_targets_dict[qed_key])]
            target_suffix += f"_{qed_key}_" + "_".join(map(str, plot_targets_dict[qed_key]))
        if logp_key in plot_targets_dict:
            plt_df = plt_df[plt_df[logp_col].isin(plot_targets_dict[logp_key])]
            target_suffix += f"_{logp_key}_" + "_".join(map(str, plot_targets_dict[logp_key]))
        if plt_df.shape[0] > 0:
            plt_df['QED-LOGP'] = plt_df[qed_col].astype(str) + '-' + plt_df[logp_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted qeds", y="Predicted logps", hue="QED-LOGP", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[qed_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[logp_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('QED vs LogP (Dual Property Conditioning)')
            caption = "Figure: QED vs LogP (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'qed_logp_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: qed_logp_kde{target_suffix}_temp_{temp}.png")
    
    # SAS vs LogP
    if 'sas' in target_props and ('logps' in target_props or 'logp' in target_props):
        sas_key = 'sas'
        logp_key = 'logps' if 'logps' in prop_names else 'logp'
        sas_col = f"Target {sas_key}"
        logp_col = f"Target {logp_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if sas_key in plot_targets_dict:
            plt_df = plt_df[plt_df[sas_col].isin(plot_targets_dict[sas_key])]
            target_suffix += f"_{sas_key}_" + "_".join(map(str, plot_targets_dict[sas_key]))
        if logp_key in plot_targets_dict:
            plt_df = plt_df[plt_df[logp_col].isin(plot_targets_dict[logp_key])]
            target_suffix += f"_{logp_key}_" + "_".join(map(str, plot_targets_dict[logp_key]))
        if plt_df.shape[0] > 0:
            plt_df['SAS-LOGP'] = plt_df[sas_col].astype(str) + '-' + plt_df[logp_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted sas", y="Predicted logps", hue="SAS-LOGP", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[sas_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[logp_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('SAS vs LogP (Dual Property Conditioning)')
            caption = "Figure: SAS vs LogP (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.savefig(os.path.join(results_dir, f'sas_logp_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: sas_logp_kde{target_suffix}_temp_{temp}.png")
    
    # Affinity vs LogP
    if 'affinity' in target_props and ('logps' in target_props or 'logp' in target_props):
        aff_key = 'affinity'
        logp_key = 'logps' if 'logps' in prop_names else 'logp'
        aff_col = f"Target {aff_key}"
        logp_col = f"Target {logp_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if aff_key in plot_targets_dict:
            plt_df = plt_df[plt_df[aff_col].isin(plot_targets_dict[aff_key])]
            target_suffix += f"_{aff_key}_" + "_".join(map(str, plot_targets_dict[aff_key]))
        if logp_key in plot_targets_dict:
            plt_df = plt_df[plt_df[logp_col].isin(plot_targets_dict[logp_key])]
            target_suffix += f"_{logp_key}_" + "_".join(map(str, plot_targets_dict[logp_key]))
        if plt_df.shape[0] > 0:
            plt_df['AFFINITY-LOGP'] = plt_df[aff_col].astype(str) + '-' + plt_df[logp_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted affinity", y="Predicted logps", hue="AFFINITY-LOGP", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[aff_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[logp_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('Binding Affinity vs LogP (Dual Property Conditioning)')
            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.ylabel('LogP')
            caption = "Figure: Binding Affinity vs LogP (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.savefig(os.path.join(results_dir, f'affinity_logp_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: affinity_logp_kde{target_suffix}_temp_{temp}.png")
    
    # Affinity vs TPSA
    if 'affinity' in target_props and ('tpsas' in target_props or 'tpsa' in target_props):
        aff_key = 'affinity'
        tpsa_key = 'tpsas' if 'tpsas' in prop_names else 'tpsa'
        aff_col = f"Target {aff_key}"
        tpsa_col = f"Target {tpsa_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if aff_key in plot_targets_dict:
            plt_df = plt_df[plt_df[aff_col].isin(plot_targets_dict[aff_key])]
            target_suffix += f"_{aff_key}_" + "_".join(map(str, plot_targets_dict[aff_key]))
        if tpsa_key in plot_targets_dict:
            plt_df = plt_df[plt_df[tpsa_col].isin(plot_targets_dict[tpsa_key])]
            target_suffix += f"_{tpsa_key}_" + "_".join(map(str, plot_targets_dict[tpsa_key]))
        if plt_df.shape[0] > 0:
            plt_df['AFFINITY-TPSA'] = plt_df[aff_col].astype(str) + '-' + plt_df[tpsa_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted affinity", y="Predicted tpsas", hue="AFFINITY-TPSA", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[aff_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[tpsa_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('Binding Affinity vs TPSA (Dual Property Conditioning)')
            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.ylabel('TPSA')
            caption = "Figure: Binding Affinity vs TPSA (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.savefig(os.path.join(results_dir, f'affinity_tpsa_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: affinity_tpsa_kde{target_suffix}_temp_{temp}.png")
    
    # Affinity vs QED
    if 'affinity' in target_props and ('qeds' in target_props or 'qed' in target_props):
        aff_key = 'affinity'
        qed_key = 'qeds' if 'qeds' in prop_names else 'qed'
        aff_col = f"Target {aff_key}"
        qed_col = f"Target {qed_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if aff_key in plot_targets_dict:
            plt_df = plt_df[plt_df[aff_col].isin(plot_targets_dict[aff_key])]
            target_suffix += f"_{aff_key}_" + "_".join(map(str, plot_targets_dict[aff_key]))
        if qed_key in plot_targets_dict:
            plt_df = plt_df[plt_df[qed_col].isin(plot_targets_dict[qed_key])]
            target_suffix += f"_{qed_key}_" + "_".join(map(str, plot_targets_dict[qed_key]))
        if plt_df.shape[0] > 0:
            plt_df['AFFINITY-QED'] = plt_df[aff_col].astype(str) + '-' + plt_df[qed_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted affinity", y="Predicted qeds", hue="AFFINITY-QED", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[aff_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[qed_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('Binding Affinity vs QED (Dual Property Conditioning)')
            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.ylabel('QED')
            caption = "Figure: Binding Affinity vs QED (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.savefig(os.path.join(results_dir, f'affinity_qed_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: affinity_qed_kde{target_suffix}_temp_{temp}.png")
    
    # Affinity vs SAS
    if 'affinity' in target_props and 'sas' in target_props:
        aff_key = 'affinity'
        sas_key = 'sas'
        aff_col = f"Target {aff_key}"
        sas_col = f"Target {sas_key}"
        plt_df = data_df.copy()
        target_suffix = ""
        if aff_key in plot_targets_dict:
            plt_df = plt_df[plt_df[aff_col].isin(plot_targets_dict[aff_key])]
            target_suffix += f"_{aff_key}_" + "_".join(map(str, plot_targets_dict[aff_key]))
        if sas_key in plot_targets_dict:
            plt_df = plt_df[plt_df[sas_col].isin(plot_targets_dict[sas_key])]
            target_suffix += f"_{sas_key}_" + "_".join(map(str, plot_targets_dict[sas_key]))
        if plt_df.shape[0] > 0:
            plt_df['AFFINITY-SAS'] = plt_df[aff_col].astype(str) + '-' + plt_df[sas_col].astype(str)
            plt.figure(figsize=(12, 8))
            ax = sns.kdeplot(data=plt_df, x="Predicted affinity", y="Predicted sas", hue="AFFINITY-SAS", 
                            fill=True, alpha=0.7, bw_adjust=1.5)
            for target_val in sorted(plt_df[aff_col].unique()):
                plt.axvline(x=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            for target_val in sorted(plt_df[sas_col].unique()):
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1, alpha=0.7)
            plt.title('Binding Affinity vs SAS (Dual Property Conditioning)')
            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.ylabel('SAS')
            caption = "Figure: Binding Affinity vs SAS (Dual Property Conditioning). Dotted lines mark the target conditions."
            plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=20)

            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.savefig(os.path.join(results_dir, f'affinity_sas_kde{target_suffix}_temp_{temp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: affinity_sas_kde{target_suffix}_temp_{temp}.png")


def main():
    checkpoint_dir = args.checkpoint_dir
    prop_names = args.properties
    temp = args.temp
     
    print("="*50)
    print(f"Analyzing molecules from: {checkpoint_dir}")
    print(f"Properties: {prop_names}")
    print(f"Temperature: {temp}")
    print("="*50)
     
    # Load and calculate properties
    results_dict = load_and_calculate_properties(checkpoint_dir, prop_names, temp)
     
    # Determine which properties were used as targets
    target_props = {}
    if 'affinity' in prop_names:
        target_props['affinity'] = True
    if 'logps' in prop_names or 'logp' in prop_names:
        target_props['logps'] = True
    if 'qeds' in prop_names or 'qed' in prop_names:
        target_props['qeds'] = True
    if 'tpsas' in prop_names or 'tpsa' in prop_names:
        target_props['tpsas'] = True
    if 'sas' in prop_names:
        target_props['sas'] = True
    
    # Create DataFrame
    data_df = create_dataframe(results_dict, prop_names)
    data_df.to_csv(os.path.join(checkpoint_dir, f'generated_data_temp_{temp}.csv'), index=False)
    print(f"\nSaved DataFrame to: {os.path.join(checkpoint_dir, f'generated_data_temp_{temp}.csv')}")
    # Compute MAE, variance, and KL divergence per-property across target-value bins
    # Load previously saved results with properties
    data_df = pd.read_csv(os.path.join(checkpoint_dir, f'generated_data_temp_{temp}.csv'))
    print("\nComputing per-property metrics (MAE, variance, KL vs Gaussian sigma=0.05)...")
    compute_all_metrics(data_df, prop_names, checkpoint_dir, temp)
    
    # Parse plot targets
    plot_targets_dict = {}
    if hasattr(args, 'plot_targets') and args.plot_targets:
        for pt in args.plot_targets:
            prop, vals = pt.split('=')
            plot_targets_dict[prop] = [float(v) for v in vals.split(',')]
            
    # Plot single property KDEs
    plot_single_property_kdes(data_df, prop_names, target_props, checkpoint_dir, temp, plot_targets_dict, args.plot_kde_separately)
    
    # Plot dual property KDEs
    plot_dual_property_kdes(data_df, prop_names, target_props, checkpoint_dir, temp, plot_targets_dict)
    
    # Generate sample molecule images
    generate_molecule_images(checkpoint_dir, temp)
    
    print(f"All results saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
