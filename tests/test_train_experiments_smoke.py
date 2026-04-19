import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_train_experiments_one_model_skips_mlflow(tmp_path):
    train = pd.read_csv(ROOT / "data/splits/train.csv").head(100)
    test = pd.read_csv(ROOT / "data/splits/test.csv").head(40)
    tr = tmp_path / "tr.csv"
    te = tmp_path / "te.csv"
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_experiments.py"),
            "--train-csv",
            str(tr),
            "--test-csv",
            str(te),
            "--tickets",
            str(ROOT / "data/processed/tickets.csv"),
            "--out-model",
            str(tmp_path / "m.joblib"),
            "--out-metrics",
            str(tmp_path / "m.json"),
            "--out-manifest",
            str(tmp_path / "manifest.json"),
            "--skip-mlflow",
            "--model-kinds",
            "logistic_regression",
        ],
        cwd=ROOT,
        check=True,
    )
    assert (tmp_path / "m.joblib").is_file()
