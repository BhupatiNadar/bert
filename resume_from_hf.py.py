from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def download_checkpoint_from_hf(
    repo_id: str,
    filename: str,
    out_dir: str | Path = "weights",
    token: str | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required.  Install it with:  pip install huggingface_hub"
        ) from exc

    token = token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "No HF token found.  Set the HF_TOKEN environment variable or "
            "pass token= explicitly."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_filename = Path(filename).name     # strip any subdirectory prefix
    local_path = out_dir / local_filename

    logger.info("Downloading  %s/%s  …", repo_id, filename)

    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
        local_dir=str(out_dir),          # save directly into out_dir
    )

    # hf_hub_download may place the file in a subfolder; make sure we
    # return a consistent path pointing to out_dir/<basename>.
    cached_path = Path(cached)
    if cached_path.name != local_path.name or cached_path.parent != out_dir:
        import shutil
        shutil.copy2(cached_path, local_path)
        logger.info("Copied to  %s", local_path)
    else:
        local_path = cached_path

    logger.info("✓ Checkpoint saved to:  %s", local_path.resolve())
    return local_path.resolve()


def resume_from_hf(
    repo_id: str,
    filename: str,
    out_dir: str | Path = "weights",
    token: str | None = None,
) -> dict:

    import torch

    local_path = download_checkpoint_from_hf(repo_id, filename, out_dir, token)

    # Peek at the checkpoint metadata without loading the full tensors
    checkpoint = torch.load(str(local_path), map_location="cpu", weights_only=False)
    epoch       = checkpoint.get("epoch",       "?")
    global_step = checkpoint.get("global_step", "?")
    loss        = checkpoint.get("loss",        "?")
    batch_idx   = checkpoint.get("batch_idx",   "?")

    tag = local_path.stem  # e.g. "tmodel_00_pct100"
    # Strip the model_filename prefix ("tmodel_") if present
    # so the tag matches what get_weights_file_path() expects.
    for prefix in ("tmodel_",):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break

    instructions = (
        f"\n{'='*60}\n"
        f"  Checkpoint summary\n"
        f"{'='*60}\n"
        f"  File        : {local_path}\n"
        f"  Epoch       : {epoch}\n"
        f"  Batch index : {batch_idx}\n"
        f"  Global step : {global_step}\n"
        f"  Loss        : {loss}\n"
        f"{'='*60}\n"
        f"  To resume training, set in config.py:\n"
        f'      "preload": "{tag}"\n'
        f"  Then run:  python train.py\n"
        f"{'='*60}\n"
    )

    logger.info(instructions)

    return {
        "local_path":   local_path,
        "tag":          tag,
        "checkpoint":   checkpoint,
        "instructions": instructions,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download a training checkpoint from a Hugging Face repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo_id", required=True,
        help='HF repo id, e.g. "username/my-nmt-checkpoints"',
    )
    p.add_argument(
        "--filename", required=True,
        help='Path of the file inside the repo, e.g. "checkpoints/tmodel_00_pct100.pt"',
    )
    p.add_argument(
        "--out_dir", default="weights",
        help="Local directory to save the checkpoint into",
    )
    p.add_argument(
        "--token", default=None,
        help="HF token (defaults to HF_TOKEN env var)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Also print resume instructions after download",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.resume:
            resume_from_hf(
                repo_id=args.repo_id,
                filename=args.filename,
                out_dir=args.out_dir,
                token=args.token,
            )
        else:
            download_checkpoint_from_hf(
                repo_id=args.repo_id,
                filename=args.filename,
                out_dir=args.out_dir,
                token=args.token,
            )
    except (EnvironmentError, ImportError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()