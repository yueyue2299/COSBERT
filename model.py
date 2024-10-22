import torch.nn as nn
import torch
import torch.nn.functional as F

class COSBERT(nn.Module):
    def __init__(self, Embedding_ChemBERT=384):
        super(COSBERT, self).__init__()

        # self.Embedding_ChemBERT = Embedding_ChemBERT # Pre-trained embeddings (E_i) from ChemBERTa-2

        # Component Embedding Network f_theta Input: E_i Output: V_COSMO, sigma profile
        self.layers = nn.Sequential(
            nn.Linear(Embedding_ChemBERT, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 52),
            nn.Softplus()
        )

    def forward(self, E_i):
        
        x = self.layers(E_i) # [B,52]

        return x