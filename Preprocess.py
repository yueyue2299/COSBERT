# SMILES converting
from Utils import get_smiles_embedding

# Numerical Operations
import numpy as np

# Reading/Writing Data
import pandas as pd

# For Progress Bar
from tqdm import tqdm

# For splitting data
from sklearn.model_selection import train_test_split

# For neglecting warnings
import warnings
from transformers import logging

def load_and_split_data(csv_file, valid_ratio, test_ratio, seed):
    data = pd.read_csv(csv_file)

    train_valid_data, test_data = train_test_split(data, test_size=test_ratio, random_state=seed)

    train_data, valid_data = train_test_split(train_valid_data, test_size=valid_ratio / (1 - test_ratio), random_state=seed)
    # train_data, valid_data = train_test_split(train_valid_data, test_size=valid_ratio, random_state=seed)

    return train_data, valid_data, test_data # [id, SMILES, Vcosmo, sigma profile], length = 54

def preprocess_and_save(data, save_path):
    smiles_list = data.iloc[:, 1].tolist()
    labels = data.iloc[:, 2:].values.astype(np.float32)
    
    embeddings = []
    for smiles in tqdm(smiles_list, desc=f'Processing {save_path}'):
        embedding = get_smiles_embedding(smiles).squeeze()  # Assuming get_smiles_embedding returns a NumPy array or tensor
        embeddings.append(embedding)
    
    embeddings = np.stack(embeddings).astype(np.float32)
    
    # Save embeddings and labels
    np.savez_compressed(save_path, embeddings=embeddings, labels=labels)

def preprocess_and_save_npz_from_csv(csv_path, seed, valid_ratio, test_ratio, storage_path='.'):
    # Suppress transformers warnings
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    
    train_data, valid_data, test_data = load_and_split_data(csv_path, valid_ratio, test_ratio, seed)

    # Preprocess and save train, validation, and test data
    train_data = pd.DataFrame(train_data)  # Assuming train_data is your pandas DataFrame
    valid_data = pd.DataFrame(valid_data)
    test_data = pd.DataFrame(test_data)

    preprocess_and_save(train_data, f'{storage_path}//train_data.npz')
    preprocess_and_save(valid_data, f'{storage_path}//valid_data.npz')
    preprocess_and_save(test_data, f'{storage_path}//test_data.npz')