import torch.nn as nn
import torch
import torch.nn.functional as F

class COSBERT(nn.Module):
    def __init__(self, Embedding_ChemBERT=384):
        super(COSBERT, self).__init__()

        # self.Embedding_ChemBERT = Embedding_ChemBERT # Pre-trained embeddings (E_i) from ChemBERTa-2
        
        # dimension of hidden layers
        hidden_dim_1 = 256
        hidden_dim_last = 128
        
        # Component Embedding Network f_theta Input: E_i Output: V_COSMO, A_COSMO, sigma profile
        
        self.shared_layers = nn.Sequential(
            nn.Linear(Embedding_ChemBERT, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_last),
            nn.ReLU()
        )
        # V_cosmo and A_cosmo output layer [2]
        self.V_A_output = nn.Sequential(
            nn.Linear(hidden_dim_last, 2)
        )
        
        # sigma profile output layer [51]
        self.sigma_output = nn.Sequential(
            nn.Linear(hidden_dim_last, 51)
            # nn.Sigmoid() # restrict the outputs between 0 and 1
        )

    def forward(self, E_i):
        shared_out = self.shared_layers(E_i)
        V_A_out = self.V_A_output(shared_out)
        sigma_out = self.sigma_output(shared_out)
        sigma_sum = torch.sum(sigma_out, dim=1, keepdim=True) + 1e-8 # prevent sum = 0
        sigma_out = sigma_out / sigma_sum # normalization
        
        return torch.cat((V_A_out, sigma_out), dim=1) # [B,53]