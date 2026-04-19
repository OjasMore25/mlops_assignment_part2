#!/usr/bin/env python3
"""Fail CI if training metrics fall below thresholds (CI/CD quality gate)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=ROOT / "models/training_metrics.json")
    parser.add_argument("--min-f1", type=float, default=0.1)
    parser.add_argument("--min-roc-auc", type=float, default=0.1)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    f1, auc = data["f1"], data["roc_auc"]
    if f1 < args.min_f1 or auc < args.min_roc_auc:
        print(f"Metric gate failed: f1={f1} roc_auc={auc}", file=sys.stderr)
        return 1
    print(f"Metric gate OK: f1={f1} roc_auc={auc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
