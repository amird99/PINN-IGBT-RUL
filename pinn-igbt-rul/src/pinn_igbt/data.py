"""Load the NASA IGBT aging data and turn raw V_CE traces into per-cycle series.

Pipeline (NumPy only, no torch):
    raw V_CE -> detect failure -> keep healthy part -> mean per cycle
             -> (optional) per-device standardisation -> (optional) EMA smoothing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DataConfig


def load_vce(raw_dir: str | Path, devices: list[str]) -> dict[str, np.ndarray]:
    """Return the concatenated collector-emitter voltage of each device's .mat file."""
    from scipy.io import loadmat  # local import keeps the rest of the module scipy-free

    raw_dir = Path(raw_dir)
    traces: dict[str, np.ndarray] = {}
    for name in devices:
        matches = sorted(raw_dir.glob(f"{name}*.mat"))  # e.g. "Device2  1.mat"
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '{name}*.mat' in {raw_dir}, found {len(matches)}. "
                "See data/README.md or run scripts/download_data.py."
            )
        mat = loadmat(matches[0], squeeze_me=True, struct_as_record=False)
        transients = mat["measurement"].transient
        traces[name] = np.concatenate([t.timeDomain.collectorEmitterVoltage for t in transients])
    return traces


def envelope(x: np.ndarray, block: int) -> np.ndarray:
    """Block-wise maximum (a trailing partial block is dropped)."""
    n = len(x) // block * block
    return x[:n].reshape(-1, block).max(axis=1)


def find_failure_index(x: np.ndarray, block: int = 10_000, frac: float = 0.8) -> int:
    """Index one past the last sample that is still at a 'healthy' V_CE level.

    A block is healthy while its maximum stays above ``frac * median(envelope)``.
    """
    env = envelope(x, block)
    threshold = frac * np.median(env)
    healthy_blocks = np.nonzero(env >= threshold)[0]
    if healthy_blocks.size == 0:
        raise ValueError("No healthy block found; check `failure_frac`.")
    last = healthy_blocks[-1]
    segment = x[last * block:(last + 1) * block]
    return int(last * block + np.nonzero(segment >= threshold)[0][-1] + 1)


def cycle_edges(n_samples: int, n_single_blocks: int, chunk: int) -> np.ndarray:
    """Sample indices separating consecutive cycles.

    The first ``n_single_blocks`` records hold one cycle each (``chunk`` samples);
    every later cycle spans ``chunk // 2`` samples.
    """
    cycle = chunk // 2
    start = n_single_blocks * chunk
    if start > n_samples:
        raise ValueError("n_single_blocks * chunk exceeds the signal length.")
    return np.concatenate([np.arange(0, start, chunk), np.arange(start, n_samples + 1, cycle)])


def cycle_means(x: np.ndarray, n_single_blocks: int, chunk: int) -> np.ndarray:
    """Average V_CE over each complete cycle (the incomplete last cycle is dropped)."""
    edges = cycle_edges(len(x), n_single_blocks, chunk)
    x = x[: edges[-1]]
    sums = np.add.reduceat(x, edges[:-1], dtype=np.float64)
    return sums / np.diff(edges)


def exponential_moving_average(x: np.ndarray, span: int) -> np.ndarray:
    """EMA identical to ``pandas.Series.ewm(span=span, adjust=False).mean()``."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def build_cycle_series(cfg: DataConfig, vce: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Per-device 1-D arrays of the cycle-averaged V_CE over the healthy life."""
    vce = vce if vce is not None else load_vce(cfg.raw_dir, cfg.devices)
    series: dict[str, np.ndarray] = {}
    for name, x in vce.items():
        failure = find_failure_index(x, cfg.envelope_block, cfg.failure_frac)
        y = cycle_means(x[:failure], cfg.n_single_blocks[name], cfg.chunk_size)
        if cfg.per_device_standardize:
            # NOTE: uses full-life statistics of the device (look-ahead at inference time).
            y = (y - y.mean()) / y.std()
        if cfg.smoothing == "ema":
            y = exponential_moving_average(y, cfg.ema_span)
        series[name] = y
    return series


def label_series(series: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Attach the normalised-RUL target (1 = new, 0 = failure) to each series."""
    return {name: (x.reshape(len(x), -1), np.linspace(1.0, 0.0, len(x))) for name, x in series.items()}
