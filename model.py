"""
BERT Model
==========
Contains:
  • Bert              – plain encoder backbone
  • BertMLMHead       – Masked Language Modeling prediction head
  • BertNSPHead       – Next Sentence Prediction classification head
  • BertForPreTraining – backbone + both heads + combined loss
  • build_bert        – factory for the raw backbone
  • build_bert_for_pretraining – factory for the full pre-training model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder_component.embedding import BertEmbedding
from encoder_component.ecoder import Encoder
from encoder_component.encoder_block import EncoderBlock
from encoder_component.multi_head_attention import MultiHeadAttentionBlock
from encoder_component.feed_forward_network import FeedForwardBlock


# ─────────────────────────────────────────────────────────────────────────────
# Core BERT Encoder Backbone
# ─────────────────────────────────────────────────────────────────────────────

class Bert(nn.Module):
    """Vanilla BERT encoder; returns the full hidden-state tensor."""

    def __init__(self, encoder: Encoder, src_embed: BertEmbedding):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.src_embed(input_ids=input_ids, token_type_ids=token_type_ids)

        # The DataLoader produces attention_mask with shape (batch, seq_len).
        # MultiHeadAttention compares it against scores of shape
        # (batch, heads, q_len, k_len), so we need to expand to
        # (batch, 1, 1, seq_len) for correct broadcasting.
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        return self.encoder(x, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Training Heads
# ─────────────────────────────────────────────────────────────────────────────

class BertMLMHead(nn.Module):
    """
    Masked Language Modeling head.

    Architecture (BERT paper §3.3.1):
        Linear(hidden_size → hidden_size) → GELU → LayerNorm → Linear(hidden_size → vocab_size)

    The final projection re-uses the token-embedding weight matrix
    (weight tying) to reduce the number of parameters, exactly as in the
    original implementation.
    """

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense      = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # The bias for the output projection is a separate learnable parameter
        self.decoder    = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias       = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, hidden_size) → (batch, seq_len, vocab_size)
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return self.decoder(x)

    def tie_weights(self, token_embedding_weight: torch.Tensor) -> None:
        """Tie the decoder matrix to the token embedding weight matrix."""
        self.decoder.weight = token_embedding_weight


class BertNSPHead(nn.Module):
    """
    Next Sentence Prediction head.

    Applies a single linear layer to the [CLS] token representation
    and returns logits for {NotNext=0, IsNext=1}.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # pooled_output: (batch, hidden_size)  →  (batch, 2)
        return self.seq_relationship(pooled_output)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pre-Training Model
# ─────────────────────────────────────────────────────────────────────────────

class BertForPreTraining(nn.Module):
    """
    BERT with both MLM and NSP heads attached.

    Forward returns a dict with keys:
        • 'loss'          – total loss (mlm_loss + nsp_loss) if labels supplied
        • 'mlm_loss'      – masked-language-model loss
        • 'nsp_loss'      – next-sentence-prediction loss
        • 'mlm_logits'    – (batch, seq_len, vocab_size)
        • 'nsp_logits'    – (batch, 2)
    """

    def __init__(self, bert: Bert, mlm_head: BertMLMHead, nsp_head: BertNSPHead):
        super().__init__()
        self.bert     = bert
        self.mlm_head = mlm_head
        self.nsp_head = nsp_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        next_sentence_label: torch.Tensor | None = None,
    ) -> dict:

        # ── Encoder pass ─────────────────────────────────────────────────────
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (batch, seq_len, hidden_size)

        # ── [CLS] pooling for NSP ────────────────────────────────────────────
        # BERT uses the [CLS] token (position 0) as the aggregate representation
        pooled_output = sequence_output[:, 0, :]  # (batch, hidden_size)

        # ── Head logits ──────────────────────────────────────────────────────
        mlm_logits = self.mlm_head(sequence_output)  # (batch, seq_len, vocab_size)
        nsp_logits = self.nsp_head(pooled_output)    # (batch, 2)

        output = {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits}

        # ── Losses (only computed when ground-truth labels are provided) ──────
        if labels is not None and next_sentence_label is not None:
            # MLM loss: only computed on the 15 % of masked positions.
            # labels == -100 at non-masked positions → ignored by CrossEntropy.
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),  # (batch*seq, vocab)
                labels.view(-1),                            # (batch*seq,)
                ignore_index=-100,
            )

            # NSP loss: standard binary cross-entropy over 2 classes
            nsp_loss = F.cross_entropy(nsp_logits, next_sentence_label)

            total_loss = mlm_loss + nsp_loss

            output["loss"]     = total_loss
            output["mlm_loss"] = mlm_loss
            output["nsp_loss"] = nsp_loss

        return output


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_bert(
    vocab_size: int,
    max_position_embeddings: int,
    d_ff: int = 3072,
    hidden_size: int = 768,
    N: int = 12,
    h: int = 12,
    dropout: float = 0.1,
) -> Bert:
    """Build the raw BERT encoder backbone."""

    assert hidden_size % h == 0, (
        "hidden_size must be divisible by the number of attention heads"
    )

    embedding = BertEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        dropout=dropout,
    )

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(
            hidden_size=hidden_size, num_heads=h, dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(
            hidden_size=hidden_size, d_ff=d_ff, dropout=dropout
        )
        encoder_block = EncoderBlock(
            self_attention_block=self_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
            hidden_size=hidden_size,
        )
        encoder_blocks.append(encoder_block)

    encoder = Encoder(layers=nn.ModuleList(encoder_blocks), hidden_size=hidden_size)
    return Bert(encoder=encoder, src_embed=embedding)


def build_bert_for_pretraining(
    vocab_size: int,
    max_position_embeddings: int,
    d_ff: int = 3072,
    hidden_size: int = 768,
    N: int = 12,
    h: int = 12,
    dropout: float = 0.1,
    layer_norm_eps: float = 1e-12,
) -> BertForPreTraining:
    """Build BERT with MLM + NSP heads, with weight tying."""

    bert = build_bert(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        d_ff=d_ff,
        hidden_size=hidden_size,
        N=N,
        h=h,
        dropout=dropout,
    )

    mlm_head = BertMLMHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layer_norm_eps=layer_norm_eps,
    )
    nsp_head = BertNSPHead(hidden_size=hidden_size)

    # Weight tying: share the token embedding matrix with the MLM decoder
    mlm_head.tie_weights(bert.src_embed.token_embedding.embedding.weight)

    model = BertForPreTraining(bert=bert, mlm_head=mlm_head, nsp_head=nsp_head)

    # ── Xavier / normal initialisation (BERT paper §3.2) ─────────────────────
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    model.apply(_init_weights)

    return model