from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food101_exp.experiment import run_experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--resume", type=str, default="none")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_experiments(args.config, args.experiment, args.resume, args.epochs)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
