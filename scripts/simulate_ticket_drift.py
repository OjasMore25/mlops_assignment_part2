#!/usr/bin/env python3
"""Emit drifted tickets to a separate CSV (does not overwrite production tickets.csv)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ticket_generator import generate_tickets_df  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/tickets_drifted.csv",
    )
    p.add_argument("--customers", type=Path, default=ROOT / "data/processed/customers.csv")
    args = p.parse_args()

    customers = pd.read_csv(args.customers)
    df = generate_tickets_df(customers, "drifted", seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote drifted tickets: {args.output} (n={len(df)})")


if __name__ == "__main__":
    main()
