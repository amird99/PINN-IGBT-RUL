"""Sliding windows, train/test splits and input scaling (NumPy only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

Arrays = tuple[np.ndarray, np.ndarray]  # (X: (n, window, features), y: (n,))


@dataclass(frozen=True)
class Scaler:
    """Per-feature z-score fitted on training windows only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        return cls(X.mean(axis=(0, 1)), X.std(axis=(0, 1)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


@dataclass
class Fold:
    """One train/test split. Data are already scaled; each device stays time-ordered."""

    name: str
    scaler: Scaler
    train: dict[str, Arrays]
    test: dict[str, Arrays]


def make_windows(x: np.ndarray, y: np.ndarray, window: int) -> Arrays:
    """Stride-1 windows; the target of a window is the label of its LAST cycle."""
    w = sliding_window_view(x, window, axis=0).transpose(0, 2, 1)
    return np.ascontiguousarray(w), y[window - 1:]


def window_all(labelled: dict[str, Arrays], window: int) -> dict[str, Arrays]:
    return {name: make_windows(x, y, window) for name, (x, y) in labelled.items()}


def _stack(parts: dict[str, Arrays]) -> Arrays:
    return (np.concatenate([x for x, _ in parts.values()]),
            np.concatenate([y for _, y in parts.values()]))


def _scale(parts: dict[str, Arrays], scaler: Scaler) -> dict[str, Arrays]:
    return {name: (scaler.transform(x), y) for name, (x, y) in parts.items()}


def in_sample_masks(n: int, group: int, block_size: int = 1, purge: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, test) masks over ``n`` consecutive windows.

    Every ``group``-th block of ``block_size`` windows is test (block_size=1 reproduces
    the original notebook). ``purge`` additionally drops training windows lying within
    ``purge`` indices of a test window; use ``purge >= window - 1`` to remove all
    windows that share a cycle with a test window.
    """
    idx = np.arange(n)
    is_test = ((idx // block_size) % group) == (group - 1)
    is_train = ~is_test
    test_idx = np.nonzero(is_test)[0]
    if purge > 0 and test_idx.size:
        distance = np.abs(idx[:, None] - test_idx[None, :]).min(axis=1)
        is_train &= distance > purge
    return is_train, is_test


def in_sample_fold(windowed: dict[str, Arrays], group: int, block_size: int, purge: int) -> Fold:
    train, test = {}, {}
    for name, (x, y) in windowed.items():
        tr, te = in_sample_masks(len(y), group, block_size, purge)
        train[name], test[name] = (x[tr], y[tr]), (x[te], y[te])
    scaler = Scaler.fit(_stack(train)[0])
    return Fold("in_sample", scaler, _scale(train, scaler), _scale(test, scaler))


def lodo_folds(windowed: dict[str, Arrays]) -> list[Fold]:
    """Leave-one-device-out: each device is the test set once; scaler fitted on the rest."""
    folds = []
    for test_name in windowed:
        train = {k: v for k, v in windowed.items() if k != test_name}
        scaler = Scaler.fit(_stack(train)[0])
        folds.append(Fold(test_name, scaler, _scale(train, scaler),
                          _scale({test_name: windowed[test_name]}, scaler)))
    return folds
