# BERT Pre-Training from Scratch 🤖

A clean, fully from-scratch PyTorch implementation of **BERT** (Bidirectional Encoder Representations from Transformers) including the complete pre-training pipeline with **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**, faithful to the original paper.

> **Paper**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Devlin et al., 2019

---

## ✨ Features

- **Full BERT encoder backbone** built from scratch (embeddings → multi-head attention → feed-forward → encoder stack)
- **Both pre-training objectives**: MLM (15% masking with 80/10/10 rule) and NSP (50% random pairs)
- **Weight tying**: MLM decoder shares weights with the token embedding matrix
- **Flexible device support**: auto-detects CPU / single GPU / 2-GPU `DataParallel`
- **Linear warmup + linear decay** LR schedule (BERT paper §A.2)
- **AdamW** with selective weight decay (biases and LayerNorm parameters are excluded)
- **Gradient clipping** to stabilise training
- **Intra-epoch checkpointing** at 25%, 50%, 75%, and 100% of each epoch
- **Resumable training**: mid-epoch resume skips already-processed batches exactly
- **Hugging Face Hub integration**: async/sync checkpoint upload with exponential-backoff retries
- **TensorBoard logging**: per-step and per-epoch loss curves for train and validation

---

## 📁 Project Structure

```
BERT/
│
├── encoder_component/           # Building blocks of the BERT encoder
│   ├── embedding.py             # BertEmbedding (token + positional + segment)
│   ├── ecoder.py                # Encoder stack (N × EncoderBlock)
│   ├── encoder_block.py         # Single encoder block (attention + FFN + residual)
│   ├── multi_head_attention.py  # Multi-head self-attention
│   ├── feed_forward_network.py  # Position-wise feed-forward network
│   ├── residual_connection.py   # Pre-LN residual wrapper
│   ├── layer_normalization.py   # Layer normalisation
│   ├── input_embedding.py       # Token embedding
│   ├── positional_embedding.py  # Learned positional embedding
│   └── segment_embedding.py     # Segment (token-type) embedding
│
├── model.py                     # Bert, BertMLMHead, BertNSPHead, BertForPreTraining + factory fns
├── dataset.py                   # BertPreTrainingDataset — MLM masking & NSP pair construction
├── train.py                     # Complete pre-training loop (CLI entry-point)
├── cofig.py                     # Hyper-parameter configuration dictionary
├── hf_uploader.py               # Async/sync HuggingFace checkpoint uploader
├── resume_from_hf.py.py         # Download & inspect a checkpoint from HF Hub
├── main.py                      # Thin wrapper over train.py
│
├── tokenizers/                  # Auto-generated WordLevel tokenizer (created on first run)
├── weights/                     # Saved checkpoints (created during training)
├── runs/                        # TensorBoard event files
│
├── requirements.txt
├── pyproject.toml
├── .env                         # HF_TOKEN and HF_REPO_ID (not committed)
└── .gitignore
```

---

## 🏗️ Model Architecture

### BERT Encoder Backbone (`Bert`)

| Component | Detail |
|---|---|
| Embedding | Token + Learned Positional + Segment embeddings |
| Encoder layers | N × (Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm) |
| Attention | Scaled dot-product, multi-head |
| Activation | GELU |
| Attention mask | Broadcasts `(batch, seq_len)` → `(batch, 1, 1, seq_len)` |

### Pre-Training Heads

| Head | Architecture |
|---|---|
| **MLM** (`BertMLMHead`) | `Linear → GELU → LayerNorm → Linear(vocab_size)` with weight tying |
| **NSP** (`BertNSPHead`) | Single `Linear(hidden_size → 2)` on the `[CLS]` token |

### Configurations

The default config (`cofig.py`) is **BERT-Tiny** for fast experimentation:

| Hyper-parameter | BERT-Tiny (default) | BERT-Base (reference) |
|---|---|---|
| `hidden_size` | 128 | 768 |
| `num_layers` | 2 | 12 |
| `num_heads` | 2 | 12 |
| `d_ff` | 512 | 3072 |
| `max_position_embeddings` | 512 | 512 |
| `seq_len` | 128 | 512 |
| `dropout` | 0.1 | 0.1 |

> **To switch to BERT-Base**, update `hidden_size`, `num_layers`, `num_heads`, and `d_ff` in `cofig.py`.

---

## 🔄 Pre-Training Objectives

### Masked Language Modeling (MLM)

- **15%** of input tokens are randomly selected as candidates.
- Of those 15%:
  - **80%** → replaced with `[MASK]`
  - **10%** → replaced with a random vocabulary token
  - **10%** → left unchanged
- Loss is computed only on the masked positions (`ignore_index=-100` everywhere else).

### Next Sentence Prediction (NSP)

- Input format: `[CLS] Sentence A [SEP] Sentence B [SEP]`
- **50%** of pairs are consecutive sentences → label `1` (IsNext)
- **50%** of pairs are random sentences from different documents → label `0` (NotNext)
- NSP uses the `[CLS]` token representation (position 0) as the aggregate sequence embedding.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install huggingface_hub   # optional — only needed for HF upload/download
```

Or using `uv`:
```bash
uv sync
```

### 2. Configure Hugging Face (Optional)

Create a `.env` file in the project root:
```env
HF_TOKEN=hf_your_token_here
HF_REPO_ID=your-username/your-repo-name
```

### 3. Run Pre-Training

```bash
# Auto-detect device (CPU / 1 GPU / 2 GPUs)
python train.py

# Force CPU
python train.py --device cpu

# Force single GPU
python train.py --device cuda

# Force 2-GPU DataParallel
python train.py --device 2gpu

# Resume from a checkpoint (e.g. epoch 0 at 50%)
python train.py --preload 0_50q

# Override epochs and batch size from CLI
python train.py --epochs 10 --batch-size 16
```

### 4. Monitor Training

```bash
tensorboard --logdir runs/
```

---

## 🗄️ Dataset

Training uses **WikiText-103** (`Salesforce/wikitext`, `wikitext-103-raw-v1`) loaded automatically via 🤗 Datasets on the first run.

- WikiText rows are grouped into per-article **documents** (separated by `=` headings / blank lines).
- Documents with fewer than 2 sentences are discarded (NSP requires at least two sentences).
- **90/10 train-validation split** is applied before training.

### Tokenizer

A **WordLevel** tokenizer is built from the training corpus on the first run and cached to `tokenizers/bert_tokenizer.json`. Special tokens: `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`. Minimum token frequency: **2**.

---

## 💾 Checkpointing

Checkpoints are saved in the `weights/` directory. The checkpoint naming convention is:

| Filename pattern | Meaning |
|---|---|
| `bmodel_0_25q.pt` | Epoch 0, 25% complete |
| `bmodel_0_50q.pt` | Epoch 0, 50% complete |
| `bmodel_0_75q.pt` | Epoch 0, 75% complete |
| `bmodel_0_100q.pt` | Epoch 0, 100% complete (end of epoch) |

Each checkpoint stores: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `epoch_int`, `step_in_epoch`, `global_step`.

### Resuming from a Checkpoint

```python
# In cofig.py
"preload": "0_50q"   # resume from epoch 0, mid-epoch at 50%
```

Or via CLI:
```bash
python train.py --preload 0_50q
```

Mid-epoch resumes automatically skip the already-processed batches using `itertools.islice`.

### Downloading a Checkpoint from HuggingFace

```bash
python resume_from_hf.py.py \
    --repo_id your-username/your-repo \
    --filename bmodel_0_100q.pt \
    --out_dir weights/ \
    --resume
```

---

## ☁️ Hugging Face Hub Integration

`HFCheckpointUploader` (`hf_uploader.py`) handles automatic checkpoint syncing:

- **Async upload** (`upload_async`): fires in a background thread so training is not blocked.
- **Sync upload** (`upload_sync`): blocks until the upload completes — used at intra-epoch milestones to guarantee safety.
- **Exponential-backoff retries**: up to 3 attempts with `2^attempt` second delays.
- **Auto repo creation**: creates the repo as private if it does not already exist.
- Upload is **disabled gracefully** if `HF_TOKEN` or `HF_REPO_ID` are not set.

---

## ⚙️ Configuration Reference

All hyper-parameters live in `cofig.py → get_config()`:

| Key | Default | Description |
|---|---|---|
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 20 | Number of training epochs |
| `seq_len` | 128 | Maximum sequence length |
| `lr` | 1e-4 | Peak learning rate |
| `adam_beta1` | 0.9 | AdamW β₁ |
| `adam_beta2` | 0.999 | AdamW β₂ |
| `adam_epsilon` | 1e-6 | AdamW ε |
| `weight_decay` | 0.01 | L2 regularisation (not applied to bias / LayerNorm) |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `warmup_steps` | 10,000 | Linear LR warmup steps |
| `mlm_probability` | 0.15 | Fraction of tokens masked |
| `hidden_size` | 128 | BERT hidden dimension |
| `num_layers` | 2 | Number of encoder layers |
| `num_heads` | 2 | Number of attention heads |
| `d_ff` | 512 | Feed-forward inner dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_position_embeddings` | 512 | Max sequence length for positional embedding |
| `layer_norm_eps` | 1e-12 | LayerNorm epsilon |
| `preload` | `None` | Checkpoint tag to resume from |
| `hf_repo_id` | env `HF_REPO_ID` | HuggingFace repository ID |
| `hf_upload` | `True` | Enable HF checkpoint upload |

---

## 📊 TensorBoard Metrics

| Tag | Description |
|---|---|
| `train/loss` | Total loss per step |
| `train/mlm_loss` | MLM loss per step |
| `train/nsp_loss` | NSP loss per step |
| `train/lr` | Learning rate per step |
| `epoch/train_loss` | Average train loss per epoch |
| `epoch/train_mlm_loss` | Average train MLM loss per epoch |
| `epoch/train_nsp_loss` | Average train NSP loss per epoch |
| `epoch/val_loss` | Average validation loss per epoch |
| `epoch/val_mlm_loss` | Average validation MLM loss per epoch |
| `epoch/val_nsp_loss` | Average validation NSP loss per epoch |

---

## 🔧 Weight Initialisation

Following the BERT paper (§3.2), all weights are initialised with:
- `Linear` and `Embedding` layers: `Normal(mean=0, std=0.02)`
- `LayerNorm` weights: `1.0`, biases: `0.0`
- Padding embedding indices: zeroed out

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `tokenizers` | Fast Rust-backed tokenizer (HuggingFace) |
| `datasets` | WikiText-103 data loading |
| `tensorboard` | Training visualisation |
| `python-dotenv` | Loading `.env` credentials |
| `huggingface_hub` | HF Hub upload/download *(optional)* |

---

## 📖 References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *NAACL-HLT 2019*.
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [HuggingFace Datasets — WikiText](https://huggingface.co/datasets/Salesforce/wikitext)

---

## 📝 License

This project is for educational and research purposes.
