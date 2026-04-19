import pandas as pd

from src.feature_engineering import build_customer_features, default_as_of_from_tickets


def test_build_customer_features_shape_and_label():
    customers = pd.read_csv("data/processed/customers.csv")
    tickets = pd.read_csv("data/processed/tickets.csv")
    as_of = default_as_of_from_tickets(tickets)
    out = build_customer_features(customers.head(50), tickets, as_of)
    assert len(out) == 50
    assert "Churn" in out.columns
    assert "charge_change_proxy" in out.columns
    assert "mean_days_between_tickets" in out.columns
