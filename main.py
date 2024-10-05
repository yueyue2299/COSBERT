# main.py

# %%
from Utils import get_embeddings, login_Hugging_for_tokenizer_model

#%%
# login to get tokenizer and model
tokenizer, model = login_Hugging_for_tokenizer_model()

#%%
def smiles_to_embeddings(smiles, tokenizer=tokenizer, model=model):
    return get_embeddings(tokenizer, model, smiles)

#%%
# Example SMILES string
smiles = "NCC"

# Get the embeddings
embedding = smiles_to_embeddings(smiles)

# Print or use the embedding as needed
print(embedding)

# %%

