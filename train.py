"""
BERT Pre-Training — Complete Training Loop
==========================================
Implements the full pre-training procedure from:
    Devlin et al., 2019 – "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding"

Device support
--------------
  • 2 GPUs  →  nn.DataParallel across cuda:0 and cuda:1
  • 1 GPU   →  single CUDA device
  • No GPU  →  CPU

Usage
-----
    python train.py                        # auto-detect device
    python train.py --device cpu           # force CPU
    python train.py --device cuda          # force single GPU
    python train.py --device 2gpu          # force DataParallel on 2 GPUs
    python train.py --preload 5            # resume from epoch 5 checkpoint
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env file before anything else so HF_TOKEN / HF_REPO_ID are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set externally

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm

from cofig import get_config, get_weight_file_path
from dataset import BertPreTrainingDataset, build_documents
from hf_uploader import HFCheckpointUploader
from model import build_bert_for_pretraining, BertForPreTraining

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bert.train")


# ─────────────────────────────────────────────────────────────────────────────
# Device Selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device_flag: Optional[str] = None) -> tuple[torch.device, bool]:
    """
    Resolve the compute device.

    Returns
    -------
    device      : the primary torch.device to move the model onto
    use_dp      : True when DataParallel over 2 GPUs should be used
    """
    if device_flag == "cpu":
        logger.info("Device: CPU (forced)")
        return torch.device("cpu"), False

    if device_flag == "2gpu":
        if torch.cuda.device_count() < 2:
            logger.warning("Requested 2 GPUs but only %d found; falling back.",
                           torch.cuda.device_count())
        else:
            logger.info("Device: DataParallel on GPU 0 + GPU 1")
            return torch.device("cuda:0"), True

    if device_flag == "cuda" or device_flag is None:
        # Auto-detect
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            logger.info("Device: DataParallel on %d GPUs", n_gpus)
            return torch.device("cuda:0"), True
        elif n_gpus == 1:
            logger.info("Device: single GPU (cuda:0)")
            return torch.device("cuda:0"), False
        else:
            logger.info("Device: CPU (no CUDA GPU found)")
            return torch.device("cpu"), False

    raise ValueError(f"Unknown --device value: {device_flag!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_all_sentences(ds):
    for item in ds:
        text = item["text"].strip()
        if text:
            yield text


def get_or_build_tokenizer(config: dict, ds) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"])
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if not tokenizer_path.exists():
        logger.info("Building tokenizer from corpus …")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        logger.info("Tokenizer saved to %s", tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info("Tokenizer loaded from %s", tokenizer_path)

    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler — linear warmup then linear decay
# ─────────────────────────────────────────────────────────────────────────────

def get_linear_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Creates a scheduler with:
        • Linear warmup from 0 → peak LR over `num_warmup_steps`
        • Linear decay from peak LR → 0 over the remaining steps

    This matches the schedule described in the BERT paper (Appendix A.2).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    config: dict,
    epoch: int | str,        # int for full-epoch saves, str like "0_25q" for quarter saves
    epoch_int: int,          # the integer epoch number (always)
    step_in_epoch: int,      # how many batches into this epoch have been processed
    global_step: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    uploader: HFCheckpointUploader,
    sync_upload: bool = False,   # True → block until HF upload finishes before returning
) -> None:
    """Save model + optimiser + scheduler state; optionally upload to HF."""
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    file_path = get_weight_file_path(config, str(epoch))

    # When using DataParallel, save the inner module weights
    model_state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )

    torch.save(
        {
            "epoch":              epoch,          # string key e.g. "0_25q" or int
            "epoch_int":          epoch_int,      # always the integer epoch index
            "step_in_epoch":      step_in_epoch,  # batches completed in this epoch
            "global_step":        global_step,
            "model_state_dict":   model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        file_path,
    )
    logger.info("Checkpoint saved → %s", file_path)

    if config.get("hf_upload"):
        if sync_upload:
            # Block training here until HF confirms the upload is complete
            logger.info("Waiting for HuggingFace upload to finish before resuming training …")
            uploader.upload_sync(file_path)
            logger.info("HuggingFace upload done — training resuming.")
        else:
            # Fire-and-forget: upload in background while training continues
            uploader.upload_async(file_path)


def load_checkpoint(
    file_path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Load a checkpoint.

    Returns
    -------
    epoch_int     : integer epoch the checkpoint belongs to
    step_in_epoch : number of batches already processed in that epoch
    global_step   : total optimizer steps taken so far
    """
    logger.info("Loading checkpoint from %s …", file_path)
    ckpt = torch.load(file_path, map_location=device)

    # Handle DataParallel wrapper
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch_int     = int(ckpt.get("epoch_int",     ckpt["epoch"]))
    step_in_epoch = int(ckpt.get("step_in_epoch", 0))
    global_step   = int(ckpt["global_step"])

    logger.info(
        "Resumed  epoch=%d  step_in_epoch=%d  global_step=%d",
        epoch_int, step_in_epoch, global_step,
    )
    return epoch_int, step_in_epoch, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, device_flag: Optional[str] = None) -> None:

    # ── Device ──────────────────────────────────────────────────────────────
    device, use_data_parallel = get_device(device_flag)
    logger.info("Torch version: %s | CUDA available: %s | GPUs: %d",
                torch.__version__,
                torch.cuda.is_available(),
                torch.cuda.device_count())

    # ── Dataset & tokenizer ──────────────────────────────────────────────────
    logger.info("Loading WikiText-103 …")
    ds_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    tokenizer = get_or_build_tokenizer(config, ds_raw)
    vocab_size = tokenizer.get_vocab_size()
    logger.info("Vocabulary size: %d", vocab_size)

    # ── Build document list for NSP ──────────────────────────────────────────
    logger.info("Building document list for NSP …")
    documents = build_documents(ds_raw)
    logger.info("Total documents: %d", len(documents))

    # ── Pre-training dataset ─────────────────────────────────────────────────
    full_dataset = BertPreTrainingDataset(
        documents=documents,
        tokenizer=tokenizer,
        seq_len=config["seq_len"],
        mlm_probability=config["mlm_probability"],
    )
    logger.info("Total pre-training examples: %d", len(full_dataset))

    # 90 / 10 train-val split
    n_val   = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model: nn.Module = build_bert_for_pretraining(
        vocab_size=vocab_size,
        max_position_embeddings=config["max_position_embeddings"],
        d_ff=config["d_ff"],
        hidden_size=config["hidden_size"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        layer_norm_eps=config["layer_norm_eps"],
    )

    # Wrap in DataParallel when two GPUs are available
    if use_data_parallel:
        gpu_ids = list(range(torch.cuda.device_count()))[:2]  # at most 2
        logger.info("Wrapping model with DataParallel on GPUs: %s", gpu_ids)
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ── Optimiser (BERT paper §A.2) ──────────────────────────────────────────
    # Do NOT apply weight-decay to bias and LayerNorm parameters
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        param_groups,
        lr=config["lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
    )

    # ── LR Scheduler ────────────────────────────────────────────────────────
    total_training_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_training_steps,
    )

    # ── TensorBoard ─────────────────────────────────────────────────────────
    writer = SummaryWriter(config["experiment_name"])

    # ── HF Uploader ──────────────────────────────────────────────────────────
    uploader = HFCheckpointUploader(repo_id=config.get("hf_repo_id"))

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    resume_step  = 0   # number of batches to skip at the start of start_epoch

    if config.get("preload") is not None:
        ckpt_path = get_weight_file_path(config, str(config["preload"]))
        if os.path.exists(ckpt_path):
            start_epoch, resume_step, global_step = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, device
            )
            if resume_step == 0:
                # Checkpoint was saved at epoch end (100 %) — start next epoch
                start_epoch += 1
            else:
                # Checkpoint was saved mid-epoch — re-enter the SAME epoch
                # and skip the already-processed batches
                logger.info(
                    "Mid-epoch resume: epoch=%d, skipping first %d batches",
                    start_epoch, resume_step,
                )
        else:
            logger.warning("Checkpoint %s not found; starting fresh.", ckpt_path)

    # ── Training Loop ────────────────────────────────────────────────────────
    logger.info(
        "Starting training: epochs=%d, steps/epoch=%d, total_steps=%d",
        config["num_epochs"],
        len(train_loader),
        total_training_steps,
    )

    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        epoch_loss     = 0.0
        epoch_mlm_loss = 0.0
        epoch_nsp_loss = 0.0

        # ── Intra-epoch checkpoint milestones (25 / 50 / 75 / 100 %) ─────────
        n_steps_epoch  = len(train_loader)
        # Step indices (1-based) at which we checkpoint — 25 %, 50 %, 75 %.
        # 100 % is handled by the existing end-of-epoch save below.
        quarter_steps  = {
            int(n_steps_epoch * 0.25): "25q",
            int(n_steps_epoch * 0.50): "50q",
            int(n_steps_epoch * 0.75): "75q",
        }
        step_in_epoch  = 0   # counts batches within this epoch

        # ── Build the batch iterator, skipping already-done steps on resume ───
        # islice(loader, resume_step, None) fast-forwards past the batches
        # that were already trained on before the checkpoint was saved.
        # After the first (resume) epoch, resume_step is reset to 0.
        raw_iter = itertools.islice(iter(train_loader), resume_step, None)
        step_in_epoch = resume_step   # start counter from where we left off
        resume_step   = 0             # only skip on the first epoch after reload

        if step_in_epoch > 0:
            logger.info(
                "Epoch %d: resuming from batch %d / %d",
                epoch + 1, step_in_epoch + 1, n_steps_epoch,
            )

        progress = tqdm(
            raw_iter,
            desc=f"Epoch {epoch + 1:>3}/{config['num_epochs']}  [train]",
            unit="batch",
            total=n_steps_epoch - step_in_epoch,
        )

        for batch in progress:
            # ── Move batch to device ─────────────────────────────────────────
            input_ids            = batch["input_ids"].to(device)
            attention_mask       = batch["attention_mask"].to(device)
            token_type_ids       = batch["token_type_ids"].to(device)
            labels               = batch["labels"].to(device)
            next_sentence_label  = batch["next_sentence_label"].to(device)

            # ── Forward pass ─────────────────────────────────────────────────
            # DataParallel returns a dict of tensors; losses are already
            # averaged across GPUs automatically.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                next_sentence_label=next_sentence_label,
            )

            # When using DataParallel, mean across the GPU dimension
            loss     = outputs["loss"].mean()
            mlm_loss = outputs["mlm_loss"].mean()
            nsp_loss = outputs["nsp_loss"].mean()

            # ── Backward pass ────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            # ── Logging ──────────────────────────────────────────────────────
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss",     loss.item(),     global_step)
            writer.add_scalar("train/mlm_loss", mlm_loss.item(), global_step)
            writer.add_scalar("train/nsp_loss", nsp_loss.item(), global_step)
            writer.add_scalar("train/lr",       current_lr,      global_step)

            epoch_loss     += loss.item()
            epoch_mlm_loss += mlm_loss.item()
            epoch_nsp_loss += nsp_loss.item()
            global_step    += 1
            step_in_epoch  += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                mlm=f"{mlm_loss.item():.4f}",
                nsp=f"{nsp_loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )

            # ── Intra-epoch checkpoint at 25 / 50 / 75 % ─────────────────────
            if step_in_epoch in quarter_steps:
                label    = quarter_steps[step_in_epoch]   # e.g. "25q"
                ckpt_key = f"{epoch}_{label}"              # e.g. "0_25q"
                logger.info(
                    "Epoch %d | %s milestone — saving checkpoint …",
                    epoch + 1, label,
                )
                save_checkpoint(
                    config=config,
                    epoch=ckpt_key,
                    epoch_int=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    uploader=uploader,
                    sync_upload=True,   # ← BLOCK until HF upload is done
                )

        # ── Epoch-level train metrics ────────────────────────────────────────
        n_batches = len(train_loader)
        avg_loss     = epoch_loss     / n_batches
        avg_mlm_loss = epoch_mlm_loss / n_batches
        avg_nsp_loss = epoch_nsp_loss / n_batches

        writer.add_scalar("epoch/train_loss",     avg_loss,     epoch)
        writer.add_scalar("epoch/train_mlm_loss", avg_mlm_loss, epoch)
        writer.add_scalar("epoch/train_nsp_loss", avg_nsp_loss, epoch)

        logger.info(
            "Epoch %d train | loss=%.4f  mlm=%.4f  nsp=%.4f",
            epoch + 1, avg_loss, avg_mlm_loss, avg_nsp_loss,
        )

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = val_mlm = val_nsp = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1:>3}  [val ]", unit="batch"):
                input_ids           = batch["input_ids"].to(device)
                attention_mask      = batch["attention_mask"].to(device)
                token_type_ids      = batch["token_type_ids"].to(device)
                labels              = batch["labels"].to(device)
                next_sentence_label = batch["next_sentence_label"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    next_sentence_label=next_sentence_label,
                )

                val_loss += outputs["loss"].mean().item()
                val_mlm  += outputs["mlm_loss"].mean().item()
                val_nsp  += outputs["nsp_loss"].mean().item()

        n_val_batches = len(val_loader)
        avg_val_loss = val_loss / n_val_batches
        avg_val_mlm  = val_mlm  / n_val_batches
        avg_val_nsp  = val_nsp  / n_val_batches

        writer.add_scalar("epoch/val_loss",     avg_val_loss, epoch)
        writer.add_scalar("epoch/val_mlm_loss", avg_val_mlm,  epoch)
        writer.add_scalar("epoch/val_nsp_loss", avg_val_nsp,  epoch)

        logger.info(
            "Epoch %d  val  | loss=%.4f  mlm=%.4f  nsp=%.4f",
            epoch + 1, avg_val_loss, avg_val_mlm, avg_val_nsp,
        )

        # ── Save checkpoint at 100 % (end of epoch) ──────────────────────────
        logger.info("Epoch %d | 100%% milestone — saving checkpoint …", epoch + 1)
        save_checkpoint(
            config=config,
            epoch=f"{epoch}_100q",   # consistent naming: e.g. "0_100q"
            epoch_int=epoch,
            step_in_epoch=0,         # 0 signals "epoch fully complete"
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            uploader=uploader,
        )

    # ── Finalise ─────────────────────────────────────────────────────────────
    writer.close()
    uploader.shutdown(wait=True)
    logger.info("Training complete. Total steps: %d", global_step)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT Pre-Training")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "2gpu"],
        default=None,
        help=(
            "Device to use: 'cpu' | 'cuda' (single GPU) | '2gpu' (DataParallel). "
            "Auto-detected if not specified."
        ),
    )
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Epoch number to resume from (e.g. --preload 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override batch size from config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = get_config()

    # Apply CLI overrides
    if args.preload is not None:
        config["preload"] = args.preload
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    train(config, device_flag=args.device)