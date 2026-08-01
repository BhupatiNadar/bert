import torch
import torch.nn as nn

from encoder_component.multi_head_attention import MultiHeadAttentionBlock
from encoder_component.residual_connection import ResidualConnection
from encoder_component.feed_forward_network import FeedForwardBlock


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, hidden_size: int):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_conection = nn.ModuleList(
            [ResidualConnection(droupot=dropout, hidden_size=hidden_size) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_conection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_conection[1](x, self.feed_forward_block)
        return x