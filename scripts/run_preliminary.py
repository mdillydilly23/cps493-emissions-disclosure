#!/usr/bin/env python3
"""Run preliminary regression + classification pipelines."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_data import resolve_data_path
from src.train_classification import train_and_evaluate as train_clf
from src.train_regression import train_and_evaluate as train_reg


def main() -> None:
    print(f"Using data: {resolve_data_path()}")
    reg = train_reg()
    print("Regression (linear):", reg["metrics"])
    clf = train_clf()
    print("Classification (logistic):", clf["metrics"])
    print("Wrote results/ and figures for your report.")


if __name__ == "__main__":
    main()
