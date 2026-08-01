import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):

    def __init__(self, hidden_size: int, d_ff: int, dropout: float):
        super().__init__()
        self.Linear1 = nn.Linear(hidden_size, d_ff)   # W1 and B1
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(d_ff, hidden_size)   # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, hidden_size) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, hidden_size)
        x = self.Linear1(x)
        x = self.activation(x)
        x = self.dropout(x)   # dropout between the two linears (BERT standard)
        x = self.Linear2(x)
        return x