import pandas as pd

from src.model_factory import build_churn_pipeline


def test_each_model_kind_fits_small_data():
    customers = pd.read_csv("data/processed/customers.csv").head(120)
    tickets = pd.read_csv("data/processed/tickets.csv")
    tickets = tickets[tickets["customer_id"].isin(customers["customer_id"])]
    from src.feature_engineering import build_customer_features, default_as_of_from_tickets, feature_columns_for_model

    as_of = default_as_of_from_tickets(tickets)
    feat = build_customer_features(customers, tickets, as_of)
    y = (feat["Churn"] == "Yes").astype(int)
    cols = feature_columns_for_model(feat)
    X = feat[cols]

    for kind in ("logistic_regression", "random_forest", "gradient_boosting"):
        pipe = build_churn_pipeline(cols, kind)
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (len(feat),)
