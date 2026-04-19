"""Ensure splits + model exist for local pytest (mirrors CI `dvc repro`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pytest_configure(config) -> None:
    """Runs early so importing `src.app` during collection finds a trained model."""
    model = ROOT / "models/churn_pipeline.joblib"
    if model.is_file():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_splits.py")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_experiments.py"),
            "--train-csv",
            str(ROOT / "data/splits/train.csv"),
            "--test-csv",
            str(ROOT / "data/splits/test.csv"),
            "--skip-mlflow",
        ],
        cwd=ROOT,
        check=True,
    )
