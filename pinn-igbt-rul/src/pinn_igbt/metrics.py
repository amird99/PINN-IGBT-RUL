"""Evaluation metrics shared by every model (NumPy only).

[FIX M2] The notebook logged plain MSE for the baseline but the *composite* PINN loss
for the PINN, so the two numbers were not comparable. Everything is reported here
with the same definitions.
"""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def monotonicity_metrics(y_pred: np.ndarray) -> dict[str, float]:
    """Violations of 'RUL never increases' on a time-ordered prediction sequence."""
    diff = np.diff(y_pred)
    if diff.size == 0:
        return {"mdc": 0.0, "violation_rate": 0.0}
    return {"mdc": float(np.mean(np.maximum(diff, 0.0) ** 2)),
            "violation_rate": float(np.mean(diff > 0))}


def boundary_metrics(y_pred: np.ndarray) -> dict[str, float]:
    """Violations of the [0, 1] range."""
    below, above = np.maximum(-y_pred, 0.0), np.maximum(y_pred - 1.0, 0.0)
    return {"bcc": float(np.mean(below ** 2) + np.mean(above ** 2)),
            "out_of_range_rate": float(np.mean((y_pred < 0) | (y_pred > 1)))}
