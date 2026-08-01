import torch
import torch.nn as nn

from encoder_component.layer_normalization import LayerNormalization


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, hidden_size: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(hidden_size=hidden_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)