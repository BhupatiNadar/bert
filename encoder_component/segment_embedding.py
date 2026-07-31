import torch
import torch.nn as nn

class SegmentEmbedding(nn.Module):
    def __init__(self, num_segments: int, hidden_size: int):
        super().__init__()

        self.segment_embedding = nn.Embedding(
            num_embeddings=num_segments,
            embedding_dim=hidden_size
        )

    def forward(self, segment_ids):
        return self.segment_embedding(segment_ids)