from pathlib import Path
import os


def get_config():
    return {
        # ── Training ─────────────────────────────────────────────────────────
        "batch_size": 8,
        "num_epochs": 20,
        "seq_len": 128,          # BERT paper pre-trains on 128 first, then 512

        # ── Optimiser (BERT paper §A) ─────────────────────────────────────────
        "lr": 1e-4,              # peak learning rate
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-6,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,    # gradient clipping

        # ── LR schedule ──────────────────────────────────────────────────────
        "warmup_steps": 10_000,  # linear warmup as per BERT paper

        # ── Pre-training objectives ───────────────────────────────────────────
        "mlm_probability": 0.15,   # fraction of tokens masked for MLM

        # ── BERT-Tiny architecture (swap to Base: 768 / 12 / 12 / 3072) ──────
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 512,
        "dropout": 0.1,

        # ── Embeddings ────────────────────────────────────────────────────────
        "max_position_embeddings": 512,
        "type_vocab_size": 2,

        # ── Model initialisation ──────────────────────────────────────────────
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,

        # ── Checkpoints ───────────────────────────────────────────────────────
        "model_folder": "weights",
        "model_filename": "bmodel_",
        "preload": None,          # set to epoch number (str) to resume

        # ── Tokenizer & logging ───────────────────────────────────────────────
        "tokenizer_file": "tokenizers/bert_tokenizer.json",
        "experiment_name": "runs/b_model",

        # ── Hugging Face ──────────────────────────────────────────────────────
        "hf_repo_id": os.environ.get("HF_REPO_ID"),
        "hf_upload": True,
    }


def get_weight_file_path(config, epoch: str) -> str:
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)