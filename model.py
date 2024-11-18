import torch.nn as nn
import torch
import torch.nn.functional as F

class COSBERT(nn.Module):
    def __init__(self, mean_VA=None, std_VA=None, Embedding_ChemBERT=384):
        super(COSBERT, self).__init__()
        self.mean_VA = mean_VA
        self.std_VA = std_VA
        # self.Embedding_ChemBERT = Embedding_ChemBERT # Pre-trained embeddings (E_i) from ChemBERTa-2
        
        # dimension of hidden layers
        hidden_dim_1 = 512
        hidden_dim_last = 256
        
        # Component Embedding Network f_theta Input: E_i Output: V_COSMO, A_COSMO, sigma profile
        
        self.shared_layers = nn.Sequential(
            nn.Linear(Embedding_ChemBERT, hidden_dim_1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_last),
            nn.Dropout(p=0.4),
            nn.ReLU()
        )
        # V_cosmo and A_cosmo output layer [2]
        self.V_A_output = nn.Sequential(
            nn.Linear(hidden_dim_last, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.LeakyReLU(negative_slope=0.01) # avoid the vanishing gradient (output is 0) problem in some situation, reduce "dead neurons"
            # x  if x > 0
            # ax if x <= 0
        )
        
        # sigma profile output layer [51]
        self.sigma_output = nn.Sequential(
            nn.Linear(hidden_dim_last, 64),
            nn.ReLU(),
            nn.Linear(64, 51)
            # nn.Sigmoid() # restrict the outputs between 0 and 1
        )

    def forward(self, E_i):
        shared_out = self.shared_layers(E_i)
        
        # Normalized output of V_cosmo and A_cosmo
        V_A_out = self.V_A_output(shared_out)
        V_A_out = (V_A_out - self.mean_VA) / self.std_VA
        
        # sigma profile
        sigma_out = self.sigma_output(shared_out)
        # sigma_out = torch.relu(sigma_out) # prevent negative values
        sigma_out = F.leaky_relu(sigma_out, negative_slope=0.01) # lelu: 0 if x<0, leaky_relu: 0 only if x=0
        sigma_sum = torch.sum(sigma_out, dim=1, keepdim=True) + 1e-8 # prevent sum = 0
        sigma_out = sigma_out / sigma_sum # normalization
        
        return torch.cat((V_A_out, sigma_out), dim=1) # [B,53]