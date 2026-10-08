"""NumPy-only tests: preprocessing, windowing, splits, scaler, metrics."""
import numpy as np

from pinn_igbt.config import DataConfig
from pinn_igbt.data import (build_cycle_series, cycle_edges, cycle_means, exponential_moving_average,
                            find_failure_index, label_series)
from pinn_igbt.metrics import boundary_metrics, monotonicity_metrics, regression_metrics
from pinn_igbt.windows import Scaler, in_sample_masks, lodo_folds, make_windows, window_all


def make_series(lengths=(60, 70, 65, 62), seed=0):
    rng = np.random.default_rng(seed)
    return {f"Device{k + 2}": 2 + 10 * k + np.cumsum(rng.normal(0.05, 0.02, n)) for k, n in enumerate(lengths)}


def test_cycle_counts_match_original_notebook():
    # (n_samples, n_single_blocks) -> cycles printed by the original notebook run.
    expected = {(13_625_000, 1): 217, (12_000_000, 1): 191, (13_000_000, 1): 207, (12_875_000, 2): 204}
    for (n, n_single), cycles in expected.items():
        assert len(cycle_edges(n, n_single, 125_000)) - 1 == cycles


def test_cycle_means_values():
    x = np.arange(22, dtype=float)
    means = cycle_means(x, n_single_blocks=1, chunk=8)  # edges 0,8,12,16,20 -> last partial dropped
    assert np.allclose(means, [x[0:8].mean(), x[8:12].mean(), x[12:16].mean(), x[16:20].mean()])


def test_find_failure_index():
    rng = np.random.default_rng(0)
    x = rng.uniform(4.5, 5.5, 6000)
    x[5037:] = rng.uniform(1.0, 2.5, 6000 - 5037)
    assert find_failure_index(x, block=100, frac=0.8) == 5037


def test_ema_matches_recursion():
    assert np.allclose(exponential_moving_average(np.array([1.0, 2.0, 3.0]), span=3), [1.0, 1.5, 2.25])


def test_window_alignment():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.linspace(1, 0, 20)
    Xw, yw = make_windows(x, y, 10)
    assert Xw.shape == (11, 10, 1) and yw.shape == (11,)
    assert np.array_equal(Xw[3, :, 0], np.arange(3, 13))
    assert yw[3] == y[12]  # target = label of the window's LAST cycle


def test_default_in_sample_split_matches_notebook_sizes():
    n_windows = [208, 182, 198, 195]  # per-device window counts in the original run
    masks = [in_sample_masks(n, group=5) for n in n_windows]
    assert sum(te.sum() for _, te in masks) == 155
    assert sum(tr.sum() for tr, _ in masks) == 628


def _leaks(train, test, window):
    ti, ei = np.nonzero(train)[0], np.nonzero(test)[0]
    return bool((np.abs(ti[:, None] - ei[None, :]) < window).any())


def test_default_split_leaks_and_purged_split_does_not():
    train, test = in_sample_masks(300, group=5)
    assert _leaks(train, test, window=10)  # documents CODE_REVIEW C2
    train, test = in_sample_masks(300, group=5, block_size=20, purge=9)
    assert train.any() and test.any()
    assert not _leaks(train, test, window=10)


def test_lodo_scaler_uses_training_devices_only():
    windowed = window_all(label_series(make_series()), window=10)
    for fold in lodo_folds(windowed):
        train_raw = np.concatenate([x for k, (x, _) in windowed.items() if k != fold.name])
        expected = Scaler.fit(train_raw)
        assert np.allclose(fold.scaler.mean, expected.mean) and np.allclose(fold.scaler.std, expected.std)
        all_raw = np.concatenate([x for x, _ in windowed.values()])
        assert not np.allclose(fold.scaler.mean, all_raw.mean(axis=(0, 1)))
        train_scaled = np.concatenate([x for x, _ in fold.train.values()])
        assert np.allclose(train_scaled.mean(), 0, atol=1e-9) and np.allclose(train_scaled.std(), 1)


def test_metrics():
    y = np.linspace(1, 0, 50)
    assert regression_metrics(y, y)["r2"] == 1.0
    assert monotonicity_metrics(y) == {"mdc": 0.0, "violation_rate": 0.0}
    assert monotonicity_metrics(y[::-1])["violation_rate"] == 1.0
    assert boundary_metrics(np.array([-0.1, 0.5, 1.2]))["out_of_range_rate"] == 2 / 3


def _synthetic_vce(n_cycles=40, chunk=16, seed=0):
    """Square-wave-like V_CE: `n_cycles` cycles of chunk//2 samples, then a collapsed tail."""
    rng = np.random.default_rng(seed)
    level = np.repeat(2.0 + 3.0 * np.linspace(0, 1, n_cycles), chunk // 2)
    healthy = level + rng.normal(0, 0.05, level.size)
    tail = rng.uniform(0.5, 1.0, 4 * chunk)
    return np.concatenate([healthy, tail])


def _cfg(**kw):
    base = dict(envelope_block=8, chunk_size=16, n_single_blocks={"D": 0}, failure_frac=0.8)
    return DataConfig(**{**base, **kw})


def test_preprocessing_chain_matches_paper_order():
    vce = {"D": _synthetic_vce()}
    raw = build_cycle_series(_cfg(per_device_standardize=False, smoothing="none"), vce)["D"]
    std = build_cycle_series(_cfg(per_device_standardize=True, smoothing="none"), vce)["D"]
    both = build_cycle_series(_cfg(per_device_standardize=True, smoothing="ema", ema_span=15), vce)["D"]
    assert len(raw) == len(std) == len(both) and 30 <= len(raw) <= 40  # failure tail was cut off
    assert np.isclose(std.mean(), 0, atol=1e-9) and np.isclose(std.std(), 1)
    assert np.allclose(both, exponential_moving_average(std, 15))      # standardise, THEN smooth
    assert np.abs(np.diff(both)).mean() < np.abs(np.diff(std)).mean()  # EMA actually smooths
