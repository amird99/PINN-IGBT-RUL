from pathlib import Path

from pinn_igbt.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_all_shipped_configs_load():
    for path in CONFIGS.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.model.hidden_size == 80 and cfg.model.head_size == 10 and cfg.data.window == 10


def test_defaults_follow_the_paper():
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.model.activation == "tanh"                      # Eq. 3
    assert cfg.data.per_device_standardize and cfg.data.smoothing == "ema" and cfg.data.ema_span == 15
    assert cfg.data.insample_group == 5                        # "for every 5 samples, the last one"


def test_paper_loss_settings():
    lodo = load_config(CONFIGS / "pinn_lodo.yaml")             # Table 2: alpha 0.1, beta 100
    assert (lodo.loss.alpha, lodo.loss.beta, lodo.loss.gamma) == (0.1, 100.0, 0.1)
    insample = load_config(CONFIGS / "pinn_in_sample.yaml")    # Table 1: alpha 0.1, beta 1
    assert (insample.loss.alpha, insample.loss.beta, insample.loss.gamma) == (0.1, 1.0, 0.1)
    assert insample.split == "in_sample" and insample.train.batching == "device_sequence"


def test_legacy_configs_reproduce_the_notebook():
    base = load_config(CONFIGS / "legacy_notebook_baseline_lodo.yaml")
    assert base.model.activation == "relu" and base.data.smoothing == "none"
    assert not base.data.per_device_standardize and base.train.batching == "minibatch"
    pinn = load_config(CONFIGS / "legacy_notebook_pinn_lodo.yaml")   # chained base: pinn_lodo -> default
    assert pinn.loss.name == "pinn" and pinn.loss.beta == 1.0 and pinn.model.activation == "relu"
    assert load_config(CONFIGS / "legacy_notebook_baseline_in_sample.yaml").train.shuffle is False


def test_overrides_and_inheritance():
    cfg = load_config(CONFIGS / "pinn_lodo.yaml", ["train.epochs=5", "loss.alpha=0.5", "seed=7",
                                                   "data.devices=[Device3]"])
    assert (cfg.train.epochs, cfg.loss.alpha, cfg.seed) == (5, 0.5, 7)
    assert cfg.data.devices == ["Device3"]
    assert cfg.loss.name == "pinn" and cfg.train.batching == "device_sequence"


def test_invalid_combinations_are_rejected():
    assert _raises(lambda: Config.from_dict({"loss": {"name": "pinn"}}))  # pinn needs ordered batches
    assert _raises(lambda: Config.from_dict({"model": {"activation": "gelu"}}))
    assert _raises(lambda: Config.from_dict({"typo_key": 1}))
    assert _raises(lambda: Config.from_dict({"train": {"epoch": 3}}))


def test_pinn_in_sample_is_allowed():
    Config.from_dict({"split": "in_sample", "loss": {"name": "pinn"}, "train": {"batching": "device_sequence"}})
