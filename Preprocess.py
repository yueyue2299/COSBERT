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

def data_normalize(csv_file):
    df = pd.read_csv(csv_file)

    Vcosmo_mean = np.mean(df['Vcosmo'].values)
    Vcosmo_normalized = df['Vcosmo'] / Vcosmo_mean
    
    density = ['-2.5', '-2.4', '-2.3', '-2.2', '-2.1', '-2.0', '-1.9', '-1.8', '-1.7', '-1.6',
                '-1.5', '-1.4', '-1.3', '-1.2', '-1.1', '-1.0', '-0.9', '-0.8', '-0.7', '-0.6',
                '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5',
                '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
                '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5']

    overall_sigma_mean = np.mean(df[density].stack())
    normal_sigma = df[density] / overall_sigma_mean
    
    data_normalized = pd.DataFrame({'id': df['id'], 'smiles': df['smiles'], 'Vcosmo': Vcosmo_normalized})
    data_normalized = pd.concat([data_normalized, normal_sigma], axis=1)
    
    return Vcosmo_mean, overall_sigma_mean, data_normalized

def load_and_split_data(csv_file, valid_ratio, test_ratio, seed, normalization=True):
    if normalization:
        Vcosmo_mean, overall_sigma_mean, data = data_normalize(csv_file)  
    # else:
    #     Vcosmo_mean, overall_sigma_mean = None, None
    #     data = pd.read_csv(csv_file)

    train_valid_data, test_data = train_test_split(data, test_size=test_ratio, random_state=seed)

    train_data, valid_data = train_test_split(train_valid_data, test_size=valid_ratio / (1 - test_ratio), random_state=seed)
    # train_data, valid_data = train_test_split(train_valid_data, test_size=valid_ratio, random_state=seed)

    for file, data in zip(['train.csv', 'valid.csv', 'test.csv'], (train_data, valid_data, test_data)):
        data.to_csv(file, index=False)
    
    return train_data, valid_data, test_data, Vcosmo_mean, overall_sigma_mean # [id, SMILES, Vcosmo, sigma profile], length = 54

def preprocess_and_save(data, save_path):
    smiles_list = data.iloc[:, 1].tolist()
    labels = data.iloc[:, 2:].values.astype(np.float32)
    
    embeddings = []
    for smiles in tqdm(smiles_list, desc=f'Processing {save_path}'):
        embedding = get_smiles_embedding(smiles).squeeze()  # Assuming get_smiles_embedding returns a NumPy array or tensor
        embeddings.append(embedding)
    
    embeddings = np.stack(embeddings).astype(np.float32)
    
    # Save smiles, embeddings and labels
    np.savez_compressed(save_path, embeddings=embeddings, labels=labels)
    

def preprocess_and_save_npz_from_csv(csv_path, seed, valid_ratio, test_ratio, storage_path='.', normalization=True):
    # Suppress transformers warnings
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    
    train_data, valid_data, test_data, Vcosmo_mean, overall_sigma_mean = load_and_split_data(csv_path, valid_ratio, test_ratio, seed, normalization)

    # Preprocess and save train, validation, and test data
    train_data = pd.DataFrame(train_data)  # Assuming train_data is your pandas DataFrame
    valid_data = pd.DataFrame(valid_data)
    test_data = pd.DataFrame(test_data)

    preprocess_and_save(train_data, f'{storage_path}//train_data.npz')
    preprocess_and_save(valid_data, f'{storage_path}//valid_data.npz')
    preprocess_and_save(test_data, f'{storage_path}//test_data.npz')
    
    return Vcosmo_mean, overall_sigma_mean
    
if __name__ == '__main__':
    df = pd.read_csv('VT2005_data_for_training.csv')
    
    Vcosmo_mean, overall_sigma_mean, data = data_normalize('VT2005_data_for_training.csv')
    print(Vcosmo_mean, overall_sigma_mean)