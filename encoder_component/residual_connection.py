import torch
import torch.nn as nn
import math

from encoder_component.layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    
    def __init__(self,droupot:float):
        super().__init__()
        self.droupout=nn.Dropout(droupot)
        self.norm=LayerNormalization()
        
    def forward(self,x,sublayer):
        return self.norm(x + self.droupout(sublayer(x)))