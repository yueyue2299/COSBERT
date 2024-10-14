# chemberta_embeddings.py

import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer
import numpy as np
from rdkit import Chem
from huggingface_hub import login

'''================================== HANNA =========================================='''

def canonicalize_smiles(smiles):
    # Canonicalize the SMILES
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smiles

def get_smiles_embedding(smiles, model_version='DeepChem/ChemBERTa-77M-MTR', max_length=128): # "DeepChem/ChemBERTa-77M-MLM"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_version, output_attentions=True) # Load ChemBERT model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_version) # Load ChemBERT tokenizer
    # Build tokenizer with same padding as in training
    tokens = tokenizer(
                smiles,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

    # Get the embeddings from CLS token
    with torch.no_grad():
        emb = model(
            tokens["input_ids"],
            tokens["attention_mask"]
        )["last_hidden_state"][:, 0, :].numpy()
    
    return emb

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''======================================================================================='''

def login_Hugging_for_tokenizer_model(key="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", model_name="DeepChem/ChemBERTa-77M-MLM"):
    # Log in to your Hugging Face account
    login(key)
    
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    
    return tokenizer, model

def get_embeddings(tokenizer, model, smiles: str):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # Using hidden states from the last layer
    return embeddings

if __name__ == '__main__':
    login_Hugging_for_tokenizer_model()