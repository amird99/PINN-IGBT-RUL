"""Typed experiment configuration, loaded from YAML with ``key=value`` overrides."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    devices: list[str] = field(default_factory=lambda: ["Device2", "Device3", "Device4", "Device5"])
    # Failure detection (block-wise envelope of V_CE).
    envelope_block: int = 10_000
    failure_frac: float = 0.8
    # Cycle segmentation: one record = `chunk_size` samples; a regular cycle = chunk_size // 2.
    chunk_size: int = 125_000
    # Number of leading records that contain ONE cycle each (determined from the dataset's record structure).
    n_single_blocks: dict[str, int] = field(
        default_factory=lambda: {"Device2": 1, "Device3": 1, "Device4": 1, "Device5": 2}
    )
    # [FIX C4] Paper preprocessing = cycle average -> standardise -> EMA(span 15). The original
    # notebook computed the last two but never used them. Defaults now follow the paper; set
    # per_device_standardize=false, smoothing=none to reproduce the notebook (configs/legacy_*).
    # The paper does not say whether standardisation is per device; here it is per device (author's choice).
    per_device_standardize: bool = True
    smoothing: str = "ema"  # "none" | "ema"
    ema_span: int = 15
    window: int = 10
    # In-sample split: every `insample_group`-th block of `insample_block_size` windows is test.
    # Defaults (block 1, purge 0) are the paper's protocol ("for every 5 samples, the last one").
    # Neighbouring windows overlap, so in-sample scores are optimistic (CODE_REVIEW C2).
    insample_group: int = 5
    insample_block_size: int = 1
    insample_purge: int = 0


@dataclass
class ModelConfig:
    input_size: int = 1
    hidden_size: int = 80
    head_size: int = 10
    # [FIX C5] Paper Eq. 3: the 10-unit layer uses tanh. The notebook used ReLU.
    activation: str = "tanh"  # "tanh" | "relu"


@dataclass
class LossConfig:
    name: str = "mse"  # "mse" | "pinn"
    alpha: float = 0.1
    beta: float = 1.0
    gamma: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-3
    batching: str = "minibatch"  # "minibatch" | "device_sequence"
    batch_size: int = 32
    eval_batch_size: int = 256
    shuffle: bool = True  # [FIX M1] notebook used shuffle=False for the in-sample baseline
    shuffle_devices: bool = False  # device_sequence mode: shuffle device order each epoch
    grad_clip: float | None = None
    log_every: int = 100


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "loss": LossConfig, "train": TrainConfig}


@dataclass
class Config:
    name: str = "experiment"
    seed: int = 42
    split: str = "lodo"  # "lodo" | "in_sample"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        _check(self.split, {"lodo", "in_sample"}, "split")
        _check(self.loss.name, {"mse", "pinn"}, "loss.name")
        _check(self.train.batching, {"minibatch", "device_sequence"}, "train.batching")
        _check(self.data.smoothing, {"none", "ema"}, "data.smoothing")
        _check(self.model.activation, {"tanh", "relu"}, "model.activation")
        if self.loss.name == "pinn" and self.train.batching != "device_sequence":
            raise ValueError(
                "loss.name='pinn' needs train.batching='device_sequence': the monotonic "
                "constraint compares consecutive cycles, so batches must be time-ordered."
            )
        if self.data.window < 1 or self.train.epochs < 1:
            raise ValueError("data.window and train.epochs must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = copy.deepcopy(raw)
        built = {key: _build(sec, raw.pop(key, None) or {}, key) for key, sec in _SECTIONS.items()}
        scalar_keys = {f.name for f in fields(cls)} - set(_SECTIONS)
        _reject_unknown(scalar_keys, raw, "top level")
        cfg = cls(**raw, **built)
        cfg.validate()
        return cfg


def _check(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is invalid; choose one of {sorted(allowed)}")


def _reject_unknown(allowed: set[str], raw: dict[str, Any], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys in {where}: {sorted(unknown)}")


def _build(cls: type, raw: dict[str, Any], where: str):
    _reject_unknown({f.name for f in fields(cls)}, raw, where)
    return cls(**raw)


def deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML (``1e-3``, ``true``)."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} must look like section.key=value")
        *parents, leaf = key.split(".")
        node = raw
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def _load_raw(path: Path) -> dict[str, Any]:
    """Read a YAML file, resolving a (possibly chained) top-level ``base:`` key."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base is not None:
        raw = deep_merge(_load_raw(path.parent / base), raw)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config; ``base: other.yaml`` inherits from (and may chain) other files."""
    return Config.from_dict(apply_overrides(_load_raw(Path(path)), overrides or []))
