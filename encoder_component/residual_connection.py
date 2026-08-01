import torch
import torch.nn as nn

from encoder_component.layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):

    def __init__(self, droupot: float, hidden_size: int):
        super().__init__()
        self.droupout = nn.Dropout(droupot)
        self.norm = LayerNormalization(hidden_size=hidden_size)

    def forward(self, x, sublayer):
        # Pre-LN: normalize first, then apply sublayer, then add residual
        return x + self.droupout(sublayer(self.norm(x)))