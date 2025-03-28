import argparse
import pandas as pd
from tqdm import tqdm

from utils import split

def build_corpus(in_path="../data/file.csv", out_path= "../data/corpus.txt"):
    smiles = pd.read_csv(in_path)['rdkitSMILES'].values
    proteins = pd.read_csv(in_path)['Target Sequence'].values
    smile_path = "../data/smiles_corpus.txt"
    protein_path = "../data/proteins_corpus.txt"

    with open(smile_path, 'w') as f:
        for sm in tqdm(smiles):
            f.write(split(sm)+'\n')
            
    with open(protein_path, 'w') as f:
        for seq in tqdm(proteins):
            line = ' '.join([P for P in seq])
            f.write(line+'\n')
    
    print('Built a corpus file!')    

# def main():
#     parser = argparse.ArgumentParser(description='Build a corpus file')
#     parser.add_argument('--in_path', '-i', type=str, default='data/chembl24_bert_train.csv', help='input file')
#     parser.add_argument('--out_path', '-o', type=str, default='data/chembl24_corpus.txt', help='output file')
#     args = parser.parse_args()

#     smiles = pd.read_csv(args.in_path)['canonical_smiles'].values
#     with open(args.out_path, 'a') as f:
#         for sm in tqdm(smiles):
#             f.write(split(sm)+'\n')
#     print('Built a corpus file!')

    
# if __name__=='__main__':
#     main()



