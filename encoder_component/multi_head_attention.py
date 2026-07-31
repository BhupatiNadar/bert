import math

import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,hidden_size: int,num_heads: int,dropout: float,):
        super().__init__()

        assert hidden_size % num_heads == 0, ("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads

        # Query, key and value projections
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        # Final output projection
        self.w_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Store attention scores for inspection
        self.attention_scores = None

    @staticmethod
    def attention(query: torch.Tensor,key: torch.Tensor,value: torch.Tensor,mask: torch.Tensor | None,dropout: nn.Dropout | None,):
        d_k = query.shape[-1]

        # query:
        # (batch, heads, query_len, d_k)
        #
        # key.transpose:
        # (batch, heads, d_k, key_len)
        #
        # attention_scores:
        # (batch, heads, query_len, key_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0,torch.finfo(attention_scores.dtype).min,)

        attention_scores = torch.softmax(attention_scores,dim=-1,)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, heads, query_len, key_len)
        # @
        # (batch, heads, key_len, d_k)
        #
        # Result:
        # (batch, heads, query_len, d_k)
        output = attention_scores @ value

        return output, attention_scores

    def forward(self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,mask: torch.Tensor | None = None,) -> torch.Tensor:
        
        batch_size = q.shape[0]
        query_len = q.shape[1]
        key_len = k.shape[1]

        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, query_len, hidden_size)
        # ->
        # (batch, query_len, heads, d_k)
        # ->
        # (batch, heads, query_len, d_k)
        query = query.view(batch_size,query_len,self.num_heads,self.d_k,).transpose(1, 2)

        # (batch, heads, key_len, d_k)
        key = key.view(batch_size,key_len,self.num_heads,self.d_k,).transpose(1, 2)

        # (batch, heads, key_len, d_k)
        value = value.view(batch_size,key_len,self.num_heads,self.d_k,).transpose(1, 2)

        x, self.attention_scores = self.attention(query=query,key=key,value=value,mask=mask,dropout=self.dropout,)

        # (batch, heads, query_len, d_k)
        # ->
        # (batch, query_len, heads, d_k)
        # ->
        # (batch, query_len, hidden_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view( batch_size,query_len,self.hidden_size,)

        return self.w_o(x)
        