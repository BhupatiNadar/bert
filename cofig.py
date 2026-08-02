from pathlib import Path
import os


def get_config():
    return {
        # Training
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 150,

        # BERT-Tiny architecture
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 512,
        "dropout": 0.1,

        # Embeddings
        "max_position_embeddings": 512,
        "type_vocab_size": 2,

        # Model initialization
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,

        # Checkpoints
        "model_folder": "weights",
        "model_filename": "bmodel_",
        "preload": None,

        # Tokenizer and logging
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/b_model",

        # Hugging Face
        "hf_repo_id": os.environ.get("HF_REPO_ID"),
        "hf_upload": True,
    }


def get_weight_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)