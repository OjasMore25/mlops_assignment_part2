"""
Shared customer-level features for training and online inference.
Uses a single reference time `as_of` so batch training and API inference stay aligned.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def default_as_of_from_tickets(tickets: pd.DataFrame, created_col: str = "created_at") -> datetime:
    return pd.to_datetime(tickets[created_col]).max().to_pydatetime()


def _mean_gap_seconds(customer_tickets: pd.DataFrame) -> float:
    s = customer_tickets.sort_values("created_at")["created_at"]
    if len(s) < 2:
        return float("nan")
    return float(s.diff().dt.total_seconds().mean())


def _gap_frame(tickets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in tickets.groupby("customer_id", sort=False):
        sec = _mean_gap_seconds(g)
        rows.append({"customer_id": cid, "mean_seconds_between_tickets": sec})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["customer_id", "mean_seconds_between_tickets"]
    )


def build_customer_features(
    customers: pd.DataFrame,
    tickets: pd.DataFrame,
    as_of: datetime,
) -> pd.DataFrame:
    """
    Build one row per customer. Only tickets with created_at <= as_of are used.
    Expects customers: customer_id, contract_type, monthly_charges, tenure, total_charges,
    and optionally Churn.
    """
    customers = customers.copy()
    tickets = tickets.copy()
    tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    tickets = tickets[tickets["created_at"] <= pd.Timestamp(as_of)]

    ticket_types = sorted(tickets["ticket_type"].dropna().unique()) if len(tickets) else []

    window_7 = as_of - timedelta(days=7)
    window_30 = as_of - timedelta(days=30)
    window_90 = as_of - timedelta(days=90)

    tickets_7 = tickets[tickets["created_at"] > window_7]
    tickets_30 = tickets[tickets["created_at"] > window_30]
    tickets_90 = tickets[tickets["created_at"] > window_90]

    f7 = tickets_7.groupby("customer_id").size().reset_index(name="tickets_last_7_days")
    f30 = tickets_30.groupby("customer_id").size().reset_index(name="tickets_last_30_days")
    f90 = tickets_90.groupby("customer_id").size().reset_index(name="tickets_last_90_days")

    complaints = tickets[tickets["ticket_type"] == "complaint"]
    complaint_feature = complaints.groupby("customer_id").size().reset_index(name="complaint_count")
    complaint_feature["complaint_ticket"] = 1
    complaint_feature = complaint_feature[["customer_id", "complaint_ticket"]]

    negative = tickets[tickets["sentiment"] == "negative"]
    neg_counts = negative.groupby("customer_id").size().reset_index(name="negative_tickets")
    total_counts = tickets.groupby("customer_id").size().reset_index(name="total_tickets")
    sentiment_feature = total_counts.merge(neg_counts, on="customer_id", how="left")
    sentiment_feature["negative_tickets"] = sentiment_feature["negative_tickets"].fillna(0)
    sentiment_feature["negative_ratio"] = sentiment_feature["negative_tickets"] / sentiment_feature[
        "total_tickets"
    ].clip(lower=1)
    sentiment_feature = sentiment_feature[["customer_id", "negative_ratio"]]

    tickets = tickets.copy()
    tickets["sentiment_score"] = tickets["sentiment"].map(SENTIMENT_SCORE).fillna(0.0)
    sentiment_score_mean = (
        tickets.groupby("customer_id")["sentiment_score"].mean().reset_index(name="sentiment_score_mean")
    )

    if len(tickets) and ticket_types:
        cat_counts = (
            tickets.groupby(["customer_id", "ticket_type"]).size().unstack(fill_value=0)
        )
        cat_counts = cat_counts.reindex(columns=ticket_types, fill_value=0)
        cat_counts = cat_counts.add_prefix("ticket_type_count_").reset_index()
    else:
        cat_counts = pd.DataFrame({"customer_id": customers["customer_id"].unique()})
        for t in ticket_types:
            cat_counts[f"ticket_type_count_{t}"] = 0

    gaps = _gap_frame(tickets)

    base_cols = ["customer_id", "contract_type", "monthly_charges", "tenure", "total_charges"]
    if "Churn" in customers.columns:
        base_cols.append("Churn")
    features = customers[base_cols].copy()

    tenure = features["tenure"].astype(float)
    implied = np.where(tenure > 0, features["total_charges"].astype(float) / tenure, np.nan)
    features["charge_change_proxy"] = features["monthly_charges"].astype(float) - np.nan_to_num(
        implied, nan=0.0
    )

    features = features.merge(f7, on="customer_id", how="left")
    features = features.merge(f30, on="customer_id", how="left")
    features = features.merge(f90, on="customer_id", how="left")
    features = features.merge(complaint_feature, on="customer_id", how="left")
    features = features.merge(sentiment_feature, on="customer_id", how="left")
    features = features.merge(sentiment_score_mean, on="customer_id", how="left")
    features = features.merge(cat_counts, on="customer_id", how="left")
    features = features.merge(gaps, on="customer_id", how="left")

    features = features.fillna(0)

    if "mean_seconds_between_tickets" in features.columns:
        features["mean_days_between_tickets"] = features["mean_seconds_between_tickets"] / 86400.0
        features = features.drop(columns=["mean_seconds_between_tickets"])
    else:
        features["mean_days_between_tickets"] = 0.0

    for t in ticket_types:
        col = f"ticket_type_count_{t}"
        if col not in features.columns:
            features[col] = 0

    return features


def feature_columns_for_model(features: pd.DataFrame) -> list[str]:
    exclude = {"customer_id", "Churn"}
    return [c for c in features.columns if c not in exclude]
