"""Synthetic ticket generation from customer churn labels (profiles in ticket_generation_profiles)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from src.ticket_generation_profiles import PROFILES

OTHER_TYPES = ["technical", "billing", "service_request", "general"]


def generate_tickets_df(
    customers: pd.DataFrame,
    profile_name: str,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)

    profile = PROFILES[profile_name]
    tickets: list[dict] = []

    for _, row in customers.iterrows():
        cid = row["customer_id"]
        churn = row["Churn"]
        p = profile.churn_yes if churn == "Yes" else profile.churn_no

        ticket_count = random.randint(p.ticket_count_min, p.ticket_count_max)

        for _ in range(ticket_count):
            if random.random() < p.complaint_prob:
                ticket_type = "complaint"
            else:
                ticket_type = random.choice(OTHER_TYPES)

            if random.random() < p.negative_prob:
                sentiment = "negative"
            else:
                sentiment = random.choice(["neutral", "positive"])

            tickets.append(
                {
                    "ticket_id": f"T{random.randint(10000, 99999)}",
                    "customer_id": cid,
                    "ticket_type": ticket_type,
                    "sentiment": sentiment,
                    "created_at": datetime.now() - timedelta(days=random.randint(1, 90)),
                }
            )

    return pd.DataFrame(tickets)
