import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    
    def __init__(self,vocab_size:int, hidden_size:int):
        super().__init__()
        
        self.embedding=nn.Embedding(
            vocab_size, hidden_size
        )
        
    def forward(self,x):
        return self.embedding(x)