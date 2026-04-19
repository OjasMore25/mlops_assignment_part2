#!/usr/bin/env python3
"""
Generate synthetic tickets from customers.csv (Churn-aware sampling).

Profiles (`reference` vs `drifted`) are defined in `src/ticket_generation_profiles.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ticket_generation_profiles import PROFILES  # noqa: E402
from src.ticket_generator import generate_tickets_df  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tickets CSV from customers.")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="reference",
        help="reference = baseline; drifted = shifted distributions for drift/CT demos",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--customers",
        type=Path,
        default=ROOT / "data/processed/customers.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/tickets.csv",
    )
    args = parser.parse_args()

    customers = pd.read_csv(args.customers)
    df = generate_tickets_df(customers, args.profile, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} tickets to {args.output} (profile={args.profile})")


if __name__ == "__main__":
    main()
