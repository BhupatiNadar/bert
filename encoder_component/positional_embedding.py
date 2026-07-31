import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):

    def __init__(self, max_position_embeddings:int, hidden_size:int):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_position_embeddings,
            hidden_size
        )

    def forward(self, position_ids):
        return self.position_embedding(position_ids)