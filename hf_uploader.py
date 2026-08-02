from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


class HFCheckpointUploader:

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID", "")
        self.token = token or os.environ.get("HF_TOKEN", "")
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

        self._enabled = bool(self.repo_id and self.token)
        self._api = None
        self._executor: ThreadPoolExecutor | None = None

        if not self._enabled:
            if not self.repo_id:
                logger.warning(
                    "[HFUploader] No repo_id supplied and HF_REPO_ID env var is unset. "
                    "HF upload is DISABLED."
                )
            if not self.token:
                logger.warning(
                    "[HFUploader] No HF token found (pass token= or set HF_TOKEN). "
                    "HF upload is DISABLED."
                )
            return

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HF checkpoint upload. "
                "Install it with:  pip install huggingface_hub"
            ) from exc

        self._api = HfApi(token=self.token)
        self._ensure_repo_exists()

        # Single worker thread: uploads are serialised so we don't saturate
        # the network while multiple checkpoints land close together.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_upload")
        logger.info("[HFUploader] Ready — repo: %s", self.repo_id)


    def upload_async(self, local_path: str | Path, repo_path: str | None = None) -> None:
        
        if not self._enabled:
            return

        local_path = Path(local_path)
        repo_path = repo_path or local_path.name

        future = self._executor.submit(
            self._upload_with_retry, str(local_path), repo_path
        )
        # Attach a callback so errors surface in the log even if no one
        # waits for the future.
        future.add_done_callback(self._log_future_exception)

    def upload_sync(self, local_path: str | Path, repo_path: str | None = None) -> None:
        """
        Upload **synchronously** — blocks the calling thread until the upload
        is fully complete (or exhausts all retries).

        Use this at training milestones where you want to guarantee the
        checkpoint is safely on HuggingFace *before* training continues.
        """
        if not self._enabled:
            return

        local_path = Path(local_path)
        repo_path = repo_path or local_path.name

        logger.info("[HFUploader] Synchronous upload starting: %s", local_path.name)
        # Submit to the serialised executor and block until it finishes.
        future = self._executor.submit(
            self._upload_with_retry, str(local_path), repo_path
        )
        # .result() re-raises any exception from the worker thread.
        try:
            future.result()
            logger.info("[HFUploader] Synchronous upload complete: %s", local_path.name)
        except Exception as exc:
            logger.error(
                "[HFUploader] Synchronous upload FAILED for %s: %s. "
                "Training will resume anyway (checkpoint saved locally).",
                local_path, exc,
            )

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            logger.info("[HFUploader] Waiting for pending uploads to finish …")
            self._executor.shutdown(wait=wait)
            logger.info("[HFUploader] All uploads complete.")


    def _ensure_repo_exists(self) -> None:
        try:
            self._api.repo_info(repo_id=self.repo_id, repo_type="model")
            logger.info("[HFUploader] Found existing repo: %s", self.repo_id)
        except Exception:
            # Repo does not exist — create it as private.
            try:
                self._api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=True,
                    exist_ok=True,
                )
                logger.info("[HFUploader] Created private repo: %s", self.repo_id)
            except Exception as exc:
                logger.error(
                    "[HFUploader] Could not create repo %s: %s. "
                    "Upload will be disabled.",
                    self.repo_id, exc,
                )
                self._enabled = False

    def _upload_with_retry(self, local_path: str, repo_path: str) -> None:
        
        for attempt in range(self.max_retries):
            try:
                self._api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=self.repo_id,
                    repo_type="model",
                )
                logger.info(
                    "[HFUploader] ✓ Uploaded  %s  →  %s/%s",
                    local_path, self.repo_id, repo_path,
                )
                return  # success — stop retrying

            except Exception as exc:
                wait = self.retry_base_seconds * (2 ** attempt)
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "[HFUploader] Upload failed (attempt %d/%d): %s. "
                        "Retrying in %.0f s …",
                        attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "[HFUploader] ✗ Upload FAILED after %d attempts for %s: %s. "
                        "Training continues — checkpoint is saved locally.",
                        self.max_retries, local_path, exc,
                    )

    @staticmethod
    def _log_future_exception(future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.error("[HFUploader] Unexpected error in upload thread: %s", exc)