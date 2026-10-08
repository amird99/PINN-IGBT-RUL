"""Re-score the checkpoints of a finished run (no retraining).

Example:
    python scripts/evaluate.py --run-dir results/pinn_lodo
"""
from __future__ import annotations

import argparse
import json

from pinn_igbt.runner import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="results/<experiment name>")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_dir, args.device)["mean_over_folds"], indent=2))


if __name__ == "__main__":
    main()
