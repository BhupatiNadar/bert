import torch
import torch.nn as nn
import math

class FeedForwardBlock(nn.Module):
    
    def __init__(self , hidden_size : int , d_ff : int , dropout : float ):
        super().__init__()
        self.Linear1=nn.Linear(hidden_size,d_ff) # W1 and B1
        self.activation=nn.GELU()
        self.Linear2=nn.Linear(d_ff,hidden_size) # w2 and b2
        self.dropout=nn.Dropout(dropout)
        
    
    def forward(self,x):
        # ( Batch , Seq_Len , hidden_size ) --> ( Batch , Seq_Len , d_ff ) --> ( Batch , Seq_Len , hidden_size )
        return self.dropout(self.Linear2(self.activation(self.Linear1(x))))