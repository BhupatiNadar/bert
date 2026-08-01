import torch
import torch.nn as nn

from encoder_component.embedding import BertEmbedding
from encoder_component.ecoder import Encoder
from encoder_component.encoder_block import EncoderBlock
from encoder_component.multi_head_attention import MultiHeadAttentionBlock
from encoder_component.feed_forward_network import FeedForwardBlock


class Bert(nn.Module):

    def __init__(self,encoder: Encoder,src_embed: BertEmbedding):
        super().__init__()

        self.encoder = encoder
        self.src_embed = src_embed

    def encode(self,input_ids: torch.Tensor,attention_mask: torch.Tensor | None = None,token_type_ids: torch.Tensor | None = None) -> torch.Tensor:

        x = self.src_embed(input_ids=input_ids,token_type_ids=token_type_ids)

        return self.encoder(x, attention_mask)

    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor | None = None,token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        
        return self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )


def build_bert(vocab_size: int,max_position_embeddings: int,d_ff: int = 3072,hidden_size: int = 768,N: int = 12,h: int = 12,dropout: float = 0.1) -> Bert:

    assert hidden_size % h == 0, (
        "hidden_size must be divisible by the number of attention heads"
    )

    embedding = BertEmbedding(vocab_size=vocab_size,hidden_size=hidden_size,max_position_embeddings=max_position_embeddings,dropout=dropout)

    encoder_blocks = []

    for _ in range(N):

        self_attention_block = MultiHeadAttentionBlock(hidden_size=hidden_size,num_heads=h,dropout=dropout)

        feed_forward_block = FeedForwardBlock(hidden_size=hidden_size,d_ff=d_ff,dropout=dropout)

        encoder_block = EncoderBlock(self_attention_block=self_attention_block, feed_forward_block=feed_forward_block, dropout=dropout, hidden_size=hidden_size)

        encoder_blocks.append(encoder_block)

    encoder = Encoder(layers=nn.ModuleList(encoder_blocks),hidden_size=hidden_size)

    bert = Bert(encoder=encoder,src_embed=embedding)

    return bert