"""Seeding, device selection and checkpoint I/O."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .models import RNNRegressor
from .windows import Scaler


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) and request deterministic kernels.

    Call it immediately before building the model so initial weights are reproducible.
    ``CUBLAS_WORKSPACE_CONFIG`` only takes effect if set before CUDA is first used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str = "auto") -> torch.device:
    """[FIX m1] Works on any torch >= 2.0 (the notebook needed torch.accelerator, >= 2.6)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Config, scaler: Scaler,
                    fold: str, history: dict[str, list[float]]) -> None:
    """[FIX M4] Weights travel with the scaler and config needed to use them."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "scaler_mean": torch.as_tensor(scaler.mean),
            "scaler_std": torch.as_tensor(scaler.std),
            "fold": fold,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """[FIX m2] Explicit map_location and weights_only (no arbitrary unpickling)."""
    return torch.load(path, map_location=device, weights_only=True)


def model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> RNNRegressor:
    model = RNNRegressor(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
