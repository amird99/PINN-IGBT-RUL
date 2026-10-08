"""Figures for training curves and predictions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(preds: dict[str, tuple[np.ndarray, np.ndarray]], path: Path | None = None,
                     ncols: int = 2):
    """True vs predicted normalised RUL, one panel per entry of ``preds``."""
    nrows = int(np.ceil(len(preds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for ax, (label, (y_true, y_pred)) in zip(axes.flat, preds.items()):
        ax.plot(y_true, label="True")
        ax.plot(y_pred, label="Predicted")
        ax.set(title=label, xlabel="Test window (time-ordered)", ylabel="Normalised RUL (1 = new, 0 = failure)")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes.flat[len(preds):]:
        ax.set_visible(False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_history(histories: dict[str, dict[str, list[float]]], path: Path | None = None, ncols: int = 2):
    """Train loss and test MSE per epoch (log scale), one panel per fold."""
    nrows = int(np.ceil(len(histories) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for ax, (label, hist) in zip(axes.flat, histories.items()):
        ax.plot(hist["train_loss"], label="train loss")
        ax.plot(hist["test_mse"], label="test MSE")
        ax.set(title=label, xlabel="Epoch", ylabel="Loss", yscale="log")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes.flat[len(histories):]:
        ax.set_visible(False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
