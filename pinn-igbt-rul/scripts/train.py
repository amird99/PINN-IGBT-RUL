"""Train and evaluate one experiment.

Example:
    python scripts/train.py --config configs/pinn_lodo.yaml
    python scripts/train.py --config configs/pinn_lodo.yaml --set train.epochs=50 --set seed=1
"""
from __future__ import annotations

import argparse

from pinn_igbt.config import load_config
from pinn_igbt.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set train.epochs=100 (repeatable).")
    args = parser.parse_args()
    run_experiment(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
