"""Pinned model checkpoints from Hugging Face."""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download as _download


def fetch(repo: str, filename: str, revision: str) -> Path:
    cache = os.environ.get("BENCHMARK_WEIGHTS", ".cache/weights")
    return Path(_download(repo_id=repo, filename=filename, revision=revision, cache_dir=cache))
