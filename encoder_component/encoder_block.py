import torch
import torch.nn as nn

from layer_normalization import LayerNormalization


class Encoder(nn.Module):

    def __init__(self, hidden_size: int, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(hidden_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)