"""Environment, seeding, device, and path resolution helpers."""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_repo_root() -> Path:
    """Return the repository root directory containing ``srp26/``."""
    return Path(__file__).resolve().parent.parent


def set_seed(seed: int = 2024) -> None:
    """Fix Python, NumPy, and PyTorch PRNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the active CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_hf_cache(default: Optional[str] = None) -> str:
    """
    Configure ``HF_HOME`` portably without relying on machine-specific paths.

    Respects an already-set ``HF_HOME`` environment variable; otherwise creates
    and returns ``default`` (or ``~/.cache/huggingface``).
    """
    if "HF_HOME" in os.environ and os.environ["HF_HOME"].strip():
        return os.environ["HF_HOME"]
    target = os.path.expanduser(default or "~/.cache/huggingface")
    os.makedirs(target, exist_ok=True)
    os.environ["HF_HOME"] = target
    return target


def resolve_data_path(dataset_name: str, repo_root: Optional[Path] = None) -> Path:
    """
    Locate a benchmark dataset file across standard repository locations.

    Checks ``dataset_name`` directly first, then ``data_sets/`` and ``data/``
    under ``SRP26_DATA_DIR`` (if set) or the repository root.
    """
    direct = Path(dataset_name)
    if direct.is_file():
        return direct

    root = Path(repo_root) if repo_root else get_repo_root()
    candidates = []
    if os.environ.get("SRP26_DATA_DIR"):
        candidates.append(Path(os.environ["SRP26_DATA_DIR"]) / dataset_name)
    candidates.extend([
        root / "data_sets" / dataset_name,
        root / "data" / dataset_name,
        Path.cwd() / "data_sets" / dataset_name,
        Path.cwd() / "data" / dataset_name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Run `uv run python data_sets/download_data.py` "
        f"or set SRP26_DATA_DIR."
    )

