import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    
    def __init__(self , hidden_size ,eps : float = 1e-12) :
        super().__init__()
        
        self.eps=eps
        
        # Learnable scale parameter (γ)
        self.alpha = nn.Parameter(torch.ones(hidden_size)) # Multiplied
        
        # Learnable shift parameter (β)
        self.bias= nn.Parameter(torch.zeros(hidden_size)) # Added
        
    def forward(self,x):
         # x shape: (batch_size, seq_len, hidden_size)
         # Mean for each token across the hidden dimension
        mean=x.mean(dim=-1,keepdim=True)
        
        # Variance for each token across the hidden dimension
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
         # Normalize, then scale and shift
        return self.alpha * (x - mean) / torch.sqrt(variance + self.eps) + self.bias