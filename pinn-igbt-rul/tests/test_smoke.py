"""End-to-end smoke tests on synthetic data (need torch; skipped otherwise)."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from pinn_igbt.config import Config  # noqa: E402
from pinn_igbt.runner import evaluate_run, run_experiment  # noqa: E402


def synthetic_series(lengths=(60, 70, 65, 62)):
    rng = np.random.default_rng(0)
    return {f"Device{k + 2}": 2 + np.cumsum(rng.normal(0.05, 0.02, n)) for k, n in enumerate(lengths)}


def make_cfg(tmp_path, **overrides):
    raw = {"name": "smoke", "device": "cpu", "output_dir": str(tmp_path),
           "train": {"epochs": 2, "log_every": 1}}
    for key, value in overrides.items():
        raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
    return Config.from_dict(raw)


@pytest.mark.parametrize("overrides", [
    {},  # baseline, mini-batch
    {"loss": {"name": "pinn"}, "train": {"epochs": 2, "batching": "device_sequence"}},
    {"split": "in_sample"},
    {"split": "in_sample", "loss": {"name": "pinn", "beta": 1.0},  # paper Table 1: PI-RNN in-sample
     "train": {"epochs": 2, "batching": "device_sequence"}},
])
def test_run_and_reevaluate(tmp_path, overrides):
    cfg = make_cfg(tmp_path, **overrides)
    series = synthetic_series()
    summary = run_experiment(cfg, series)
    run_dir = tmp_path / "smoke"
    assert (run_dir / "metrics.json").exists() and list((run_dir / "checkpoints").glob("*.pt"))
    assert np.isfinite(summary["mean_over_folds"]["mse"])
    again = evaluate_run(run_dir, "cpu", series)
    assert again["mean_over_folds"]["mse"] == pytest.approx(summary["mean_over_folds"]["mse"], rel=1e-6)
    json.loads((run_dir / "metrics.json").read_text())
