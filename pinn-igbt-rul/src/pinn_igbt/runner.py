"""End-to-end experiments: data -> folds -> train -> evaluate -> save."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import build_cycle_series, label_series
from .engine import Batch, fit, predict, to_tensor
from .losses import build_loss
from .metrics import boundary_metrics, monotonicity_metrics, regression_metrics
from .models import RNNRegressor
from .plotting import plot_history, plot_predictions
from .utils import load_checkpoint, model_from_checkpoint, resolve_device, save_checkpoint, set_seed
from .windows import Fold, in_sample_fold, lodo_folds, window_all


def prepare_folds(cfg: Config, series: dict[str, np.ndarray] | None = None) -> list[Fold]:
    """Build the (scaled) train/test folds. ``series`` lets tests inject synthetic data."""
    series = series if series is not None else build_cycle_series(cfg.data)
    windowed = window_all(label_series(series), cfg.data.window)
    if cfg.split == "lodo":
        return lodo_folds(windowed)
    d = cfg.data
    return [in_sample_fold(windowed, d.insample_group, d.insample_block_size, d.insample_purge)]


def _train_batches(cfg: Config, fold: Fold) -> Callable[[], Iterable[Batch]]:
    """Return a factory producing one epoch of training batches."""
    if cfg.train.batching == "minibatch":
        X = np.concatenate([x for x, _ in fold.train.values()])
        y = np.concatenate([t for _, t in fold.train.values()])
        loader = DataLoader(
            TensorDataset(to_tensor(X), to_tensor(y)),
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle,
            generator=torch.Generator().manual_seed(cfg.seed),  # reproducible shuffling
        )
        return lambda: loader

    # device_sequence: one ordered batch per training device (needed by the MDC term).
    sequences = [(to_tensor(x), to_tensor(y)) for x, y in fold.train.values()]
    rng = np.random.default_rng(cfg.seed)

    def epoch_batches() -> list[Batch]:
        order = rng.permutation(len(sequences)) if cfg.train.shuffle_devices else range(len(sequences))
        return [sequences[i] for i in order]

    return epoch_batches


def score_fold(model: torch.nn.Module, fold: Fold, cfg: Config, device: torch.device):
    """Metrics and predictions for every test device of a fold."""
    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (X, y) in fold.test.items():
        y_pred = predict(model, to_tensor(X), device, cfg.train.eval_batch_size)
        # Predictions are time-ordered; for in-sample the test points are spaced apart, but
        # RUL must still never increase between them, so the MDC metric stays meaningful.
        m = {**regression_metrics(y, y_pred), **boundary_metrics(y_pred), **monotonicity_metrics(y_pred)}
        metrics[name], preds[name] = m, (y, y_pred)
    if len(preds) > 1:
        metrics["ALL"] = regression_metrics(np.concatenate([p[0] for p in preds.values()]),
                                            np.concatenate([p[1] for p in preds.values()]))
    return metrics, preds


def _mean_over_folds(folds_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    rows = [m for fold in folds_metrics.values() for key, m in fold.items() if key != "ALL"]
    keys = set.intersection(*(set(r) for r in rows))
    return {k: float(np.mean([r[k] for r in rows])) for k in sorted(keys)}


def run_experiment(cfg: Config, series: dict[str, np.ndarray] | None = None) -> dict:
    """Train one model per fold, save checkpoints / metrics / figures under results/<name>/."""
    cfg.validate()
    device = resolve_device(cfg.device)
    run_dir = Path(cfg.output_dir) / cfg.name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"[{cfg.name}] device={device} split={cfg.split} loss={cfg.loss.name} "
          f"batching={cfg.train.batching}")

    all_metrics, all_preds, histories = {}, {}, {}
    for fold in prepare_folds(cfg, series):
        print(f"--- fold: {fold.name} ---")
        set_seed(cfg.seed)  # right before model creation -> same init as the notebook's manual_seed(42)
        model = RNNRegressor(**asdict(cfg.model)).to(device)

        X_test = np.concatenate([x for x, _ in fold.test.values()])
        y_test = np.concatenate([y for _, y in fold.test.values()])
        history = fit(model, _train_batches(cfg, fold), to_tensor(X_test), y_test,
                      build_loss(cfg.loss), cfg.train, device)

        metrics, preds = score_fold(model, fold, cfg, device)
        all_metrics[fold.name], histories[fold.name] = metrics, history
        for dev, pair in preds.items():
            all_preds[dev if cfg.split == "in_sample" else fold.name] = pair
        save_checkpoint(run_dir / "checkpoints" / f"{fold.name}.pt", model, cfg, fold.scaler,
                        fold.name, history)

    summary = {"name": cfg.name, "folds": all_metrics, "mean_over_folds": _mean_over_folds(all_metrics)}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(run_dir / "predictions.npz", **{f"{k}__{t}": v[i] for k, v in all_preds.items()
                                             for i, t in enumerate(("true", "pred"))})
    plot_predictions(all_preds, run_dir / "predictions.png")
    plot_history(histories, run_dir / "history.png")
    plt.close("all")
    print(f"[{cfg.name}] mean over folds: " +
          ", ".join(f"{k}={v:.4g}" for k, v in summary["mean_over_folds"].items()
                    if k in ("mse", "rmse", "mae", "r2", "mdc")))
    return summary


def evaluate_run(run_dir: str | Path, device_name: str = "auto",
                 series: dict[str, np.ndarray] | None = None) -> dict:
    """Re-score saved checkpoints without retraining.

    The data pipeline is rebuilt from the stored config; its scaler must match the one saved
    with each checkpoint, otherwise inputs would be normalised inconsistently.
    """
    run_dir = Path(run_dir)
    device = resolve_device(device_name)
    paths = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir / 'checkpoints'}")
    cfg = Config.from_dict(load_checkpoint(paths[0], device)["config"])
    folds = {f.name: f for f in prepare_folds(cfg, series)}

    all_metrics, all_preds = {}, {}
    for path in paths:
        ckpt = load_checkpoint(path, device)
        fold = folds[ckpt["fold"]]
        if not (np.allclose(fold.scaler.mean, ckpt["scaler_mean"].cpu().numpy())
                and np.allclose(fold.scaler.std, ckpt["scaler_std"].cpu().numpy())):
            raise RuntimeError(f"{path.name}: rebuilt scaler differs from the checkpoint's; "
                               "the data pipeline changed since training.")
        metrics, preds = score_fold(model_from_checkpoint(ckpt, device), fold, cfg, device)
        all_metrics[fold.name] = metrics
        for dev, pair in preds.items():
            all_preds[dev if cfg.split == "in_sample" else fold.name] = pair

    summary = {"name": cfg.name, "folds": all_metrics, "mean_over_folds": _mean_over_folds(all_metrics)}
    (run_dir / "metrics_eval.json").write_text(json.dumps(summary, indent=2))
    plot_predictions(all_preds, run_dir / "predictions_eval.png")
    plt.close("all")
    return summary
