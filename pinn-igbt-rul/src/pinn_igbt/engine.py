"""Training and inference loops."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

from .config import TrainConfig

Batch = tuple[torch.Tensor, torch.Tensor]


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32)


def train_epoch(model: nn.Module, batches: Iterable[Batch], optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device,
                grad_clip: float | None = None) -> dict[str, float]:
    """One pass over ``batches``; returns sample-weighted means of the loss and its parts.

    [FIX M3] The notebook reported the loss parts of the LAST batch only.
    """
    model.train()
    totals: dict[str, float] = defaultdict(float)
    n_total = 0
    for X, y in batches:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X).squeeze(-1)  # (B, 1) -> (B,), same shape as y
        loss, parts = criterion(y_hat, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = y.shape[0]
        n_total += n
        totals["loss"] += loss.item() * n
        for key, value in parts.items():
            totals[key] += float(value) * n
    return {key: value / n_total for key, value in totals.items()}


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Predictions as a flat NumPy array (model is switched to eval mode)."""
    model.eval()
    outputs = [model(X[i:i + batch_size].to(device)).squeeze(-1).cpu()
               for i in range(0, len(X), batch_size)]
    return torch.cat(outputs).numpy()


def fit(model: nn.Module, make_batches: Callable[[], Iterable[Batch]], X_test: torch.Tensor,
        y_test: np.ndarray, criterion: nn.Module, cfg: TrainConfig,
        device: torch.device) -> dict[str, list[float]]:
    """Train with Adam. The test set is only *monitored* (no early stopping / selection)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: dict[str, list[float]] = defaultdict(list)

    for epoch in range(1, cfg.epochs + 1):
        stats = train_epoch(model, make_batches(), optimizer, criterion, device, cfg.grad_clip)
        test_mse = float(np.mean((predict(model, X_test, device, cfg.eval_batch_size) - y_test) ** 2))
        for key, value in stats.items():
            history[f"train_{key}"].append(value)
        history["test_mse"].append(test_mse)

        if epoch == 1 or epoch % cfg.log_every == 0:
            parts = ", ".join(f"{k}={v:.2e}" for k, v in stats.items() if k != "loss")
            print(f"epoch {epoch:>5}/{cfg.epochs} | train loss {stats['loss']:.3e} ({parts}) "
                  f"| test mse {test_mse:.3e}")
    return dict(history)
