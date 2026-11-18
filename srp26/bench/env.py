"""Seeding, determinism, dataset verification and run provenance (protocol §2)."""

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SRP26_DATA_DIR", REPO_ROOT / "data_sets"))
CHECKSUM_FILE = REPO_ROOT / "data_sets" / "CHECKSUMS.sha256"


def seed_everything(seed: int) -> torch.Generator:
    """Seed every RNG and enable deterministic kernels (best effort, protocol §2.3).

    Returns a ``torch.Generator`` to drive DataLoader shuffling.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def expected_checksums() -> Dict[str, str]:
    out = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest
    return out


_VERIFIED: Dict[str, str] = {}


def verify_dataset(filename: str) -> str:
    """Refuse to run on a dataset whose SHA-256 differs from the recorded one."""
    if filename in _VERIFIED:
        return _VERIFIED[filename]
    expected = expected_checksums().get(filename)
    if expected is None:
        raise RuntimeError(f"No recorded checksum for {filename} in {CHECKSUM_FILE}.")
    actual = sha256_file(DATA_DIR / filename)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {filename}: expected {expected}, got {actual}.")
    _VERIFIED[filename] = actual
    return actual


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_state() -> Dict[str, object]:
    """Commit hash and whether tracked code files are modified (results/docs ignored)."""
    commit = os.environ.get("SRP26_GIT_COMMIT") or _git("rev-parse", "HEAD")
    if "SRP26_GIT_COMMIT" in os.environ:  # cluster copy is an exported archive, not a checkout
        return {"commit": commit, "dirty": os.environ.get("SRP26_GIT_DIRTY", "0") == "1"}
    status = _git("status", "--porcelain", "--", "srp26", "models", "configs", "pyproject.toml", "uv.lock")
    return {"commit": commit, "dirty": bool(status) and status != "unknown"}


def provenance(device: torch.device) -> Dict[str, object]:
    import scipy
    import pandas

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return {
        **git_state(),
        "host": os.environ.get("SRP26_HOST_LABEL", platform.node()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "gpu": gpu,
        "platform": platform.platform(),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
