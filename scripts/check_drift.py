#!/usr/bin/env python3
"""
Lightweight data-drift check: KS distance on numeric feature columns vs reference split.
Part 2 PDF — data drift detection (MLOps). Writes JSON report; optional non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    build_customer_features,
    default_as_of_from_tickets,
    feature_columns_for_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "data/splits/train.csv")
    parser.add_argument("--customers", type=Path, default=ROOT / "data/processed/customers.csv")
    parser.add_argument(
        "--tickets",
        type=Path,
        default=ROOT / "data/processed/tickets.csv",
        help="Current ticket log to compare (use tickets_drifted.csv after simulate_ticket_drift.py)",
    )
    parser.add_argument("--out-report", type=Path, default=ROOT / "models/drift_report.json")
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.12,
        help="Flag drift if KS stat exceeds this (lower = stricter; drifted profile often >0.15 on ticket features)",
    )
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    if not args.reference.is_file():
        print("No reference split; skip drift check.", file=sys.stderr)
        return 0

    ref = pd.read_csv(args.reference)
    customers = pd.read_csv(args.customers)
    tickets = pd.read_csv(args.tickets)
    as_of = default_as_of_from_tickets(tickets)
    cur = build_customer_features(customers, tickets, as_of)

    cols = [c for c in feature_columns_for_model(ref) if c in ref.columns and c in cur.columns]
    numeric = [c for c in cols if c != "contract_type"]

    rows = []
    drifted = False
    for col in numeric:
        a = ref[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        b = cur[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ks = float(stats.ks_2samp(a, b, method="auto").statistic)
        flagged = ks > args.ks_threshold
        drifted = drifted or flagged
        rows.append({"column": col, "ks_statistic": ks, "drift_flag": flagged})

    report = {"columns": rows, "any_drift": drifted}
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.fail_on_drift and drifted:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
