#!/usr/bin/env python3
"""
Create stratified train / val / test CSVs (Part 2 PDF — training splits under data versioning).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    build_customer_features,
    default_as_of_from_tickets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=Path, default=ROOT / "data/processed/customers.csv")
    parser.add_argument("--tickets", type=Path, default=ROOT / "data/processed/tickets.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    customers = pd.read_csv(args.customers)
    tickets = pd.read_csv(args.tickets)
    as_of = default_as_of_from_tickets(tickets)
    features = build_customer_features(customers, tickets, as_of)
    y = (features["Churn"] == "Yes").astype(int)

    # Hold out test, then split remainder into train / val (val is val_size of full population)
    train_val, test_df = train_test_split(
        features,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    y_tv = (train_val["Churn"] == "Yes").astype(int)
    val_fraction = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=y_tv,
        random_state=args.random_state,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)
    print(
        f"Wrote splits to {args.out_dir}: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


if __name__ == "__main__":
    main()
