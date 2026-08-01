import torch
import torch.nn as nn

from encoder_component.input_embedding import InputEmbedding
from encoder_component.positional_embedding import PositionEmbedding
from encoder_component.segment_embedding import SegmentEmbedding


class BertEmbedding(nn.Module):

    def __init__(self,vocab_size: int,hidden_size: int,max_position_embeddings: int,num_segments: int = 2,dropout: float = 0.1):
        super().__init__()

        self.token_embedding = InputEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        self.position_embedding = PositionEmbedding(
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
        )

        self.segment_embedding = SegmentEmbedding(
            max_segment_embedding=num_segments,
            hidden_size=hidden_size,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self,input_ids: torch.Tensor,token_type_ids: torch.Tensor | None = None,) -> torch.Tensor:

        batch_size, sequence_length = input_ids.shape

        # Shape: (1, sequence_length)
        position_ids = torch.arange(sequence_length,device=input_ids.device,dtype=torch.long,).unsqueeze(0)

        # Shape: (batch_size, sequence_length)
        position_ids = position_ids.expand(batch_size, sequence_length)

        # If segment IDs are not supplied, all tokens belong to segment 0
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Each output shape:
        # (batch_size, sequence_length, hidden_size)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)

        # BERT combines embeddings using element-wise addition
        embeddings = (token_embeddings + position_embeddings + segment_embeddings)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
        