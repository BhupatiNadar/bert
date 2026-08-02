"""
BERT Pre-Training Dataset
=========================
Implements both pre-training objectives from the original BERT paper:

    Devlin et al., 2019 – "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding"

Masked Language Modeling (MLM)
-------------------------------
  • 15 % of tokens are selected as candidates.
  • Of those 15 %:
      – 80 % are replaced with [MASK]
      – 10 % are replaced with a random vocabulary token
      – 10 % are left unchanged
  • Loss is computed only on the selected 15 %.

Next Sentence Prediction (NSP)
-------------------------------
  • 50 % of pairs are consecutive sentences (label = 1  /  IsNext)
  • 50 % of pairs are random sentences      (label = 0  /  NotNext)
  • Format: [CLS] sentence_A [SEP] sentence_B [SEP]
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Helper: split a HuggingFace dataset into per-document sentence lists
# ──────────────────────────────────────────────────────────────────────────────

def build_documents(ds_raw) -> List[List[str]]:
    """
    Group raw WikiText rows into documents (separated by blank lines /
    article headings that start with ' = ').

    Returns a list of documents, where each document is a list of
    non-empty sentence strings.
    """
    documents: List[List[str]] = []
    current_doc: List[str] = []

    for item in ds_raw:
        text = item["text"].strip()

        # WikiText article headers start with ' = Title = '
        if text.startswith("=") or text == "":
            if current_doc:
                documents.append(current_doc)
                current_doc = []
        else:
            current_doc.append(text)

    if current_doc:
        documents.append(current_doc)

    # Drop single-sentence documents (cannot form an NSP pair)
    documents = [doc for doc in documents if len(doc) >= 2]
    return documents


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BertPreTrainingDataset(Dataset):
    """
    Builds (input_ids, attention_mask, token_type_ids, labels,
    next_sentence_label) tuples suitable for BERT pre-training.

    Parameters
    ----------
    documents       : list of documents, each being a list of sentence strings
    tokenizer       : HuggingFace `tokenizers.Tokenizer` instance
    seq_len         : maximum total sequence length (default 128)
    mlm_probability : fraction of tokens to mask (default 0.15)
    """

    def __init__(
        self,
        documents: List[List[str]],
        tokenizer,
        seq_len: int = 128,
        mlm_probability: float = 0.15,
    ) -> None:
        super().__init__()

        self.documents = documents
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mlm_probability = mlm_probability

        # Cache special token IDs
        vocab = tokenizer.get_vocab()
        self.cls_id   = vocab["[CLS]"]
        self.sep_id   = vocab["[SEP]"]
        self.pad_id   = vocab["[PAD]"]
        self.mask_id  = vocab["[MASK]"]
        self.vocab_size = tokenizer.get_vocab_size()

        # Build the flat list of (sentence_A, sentence_B, is_next) examples
        self.examples = self._build_examples()

    # ── Internal builders ────────────────────────────────────────────────────

    def _build_examples(self) -> List[Dict]:
        """
        Create one NSP example per sentence pair.
        Half are consecutive (is_next=1), half are random (is_next=0).
        """
        examples = []

        for doc_idx, document in enumerate(self.documents):
            for sent_idx in range(len(document) - 1):
                sent_a = document[sent_idx]

                # 50 % chance of a random (NotNext) sentence
                if random.random() < 0.5:
                    sent_b = document[sent_idx + 1]
                    is_next = 1
                else:
                    # Pick a random document (different from current)
                    rand_doc_idx = doc_idx
                    while rand_doc_idx == doc_idx:
                        rand_doc_idx = random.randint(0, len(self.documents) - 1)
                    rand_sent_idx = random.randint(
                        0, len(self.documents[rand_doc_idx]) - 1
                    )
                    sent_b = self.documents[rand_doc_idx][rand_sent_idx]
                    is_next = 0

                examples.append(
                    {"sent_a": sent_a, "sent_b": sent_b, "is_next": is_next}
                )

        return examples

    def _tokenize(self, text: str) -> List[int]:
        """Return a list of token IDs for a plain text sentence."""
        return self.tokenizer.encode(text).ids

    def _apply_mlm(
        self, token_ids: List[int]
    ) -> tuple[List[int], List[int]]:
        """
        Apply MLM masking to a list of token IDs.

        Returns
        -------
        masked_ids : token IDs with masking applied
        labels     : -100 everywhere except at masked positions
                     (so CrossEntropyLoss ignores non-masked tokens)
        """
        masked_ids = list(token_ids)
        labels = [-100] * len(token_ids)

        # Special tokens ([CLS], [SEP], [PAD]) are never masked
        special = {self.cls_id, self.sep_id, self.pad_id}

        for i, tok_id in enumerate(token_ids):
            if tok_id in special:
                continue

            if random.random() < self.mlm_probability:
                labels[i] = tok_id  # record original token for loss

                r = random.random()
                if r < 0.80:
                    masked_ids[i] = self.mask_id          # 80 % → [MASK]
                elif r < 0.90:
                    masked_ids[i] = random.randint(        # 10 % → random tok
                        0, self.vocab_size - 1
                    )
                # else: 10 % → keep original (masked_ids[i] unchanged)

        return masked_ids, labels

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        ids_a = self._tokenize(example["sent_a"])
        ids_b = self._tokenize(example["sent_b"])

        # ── Truncate so that the full sequence fits within seq_len ───────────
        # Budget = seq_len - 3  (for [CLS], [SEP], [SEP])
        budget = self.seq_len - 3
        # Truncate the longer side first (following BERT paper)
        while len(ids_a) + len(ids_b) > budget:
            if len(ids_a) > len(ids_b):
                ids_a.pop()
            else:
                ids_b.pop()

        # ── Build input sequence: [CLS] A [SEP] B [SEP] ─────────────────────
        tokens = (
            [self.cls_id]
            + ids_a
            + [self.sep_id]
            + ids_b
            + [self.sep_id]
        )

        # Segment IDs: 0 for [CLS]+A+[SEP], 1 for B+[SEP]
        token_type_ids = (
            [0] * (len(ids_a) + 2)   # [CLS] + A + [SEP]
            + [1] * (len(ids_b) + 1) # B + [SEP]
        )

        # ── Padding ──────────────────────────────────────────────────────────
        pad_len = self.seq_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len
        tokens = tokens + [self.pad_id] * pad_len

        # ── Apply MLM ────────────────────────────────────────────────────────
        masked_tokens, mlm_labels = self._apply_mlm(tokens)

        return {
            "input_ids":           torch.tensor(masked_tokens,  dtype=torch.long),
            "attention_mask":      torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids":      torch.tensor(token_type_ids, dtype=torch.long),
            "labels":              torch.tensor(mlm_labels,     dtype=torch.long),
            "next_sentence_label": torch.tensor(example["is_next"], dtype=torch.long),
        }