import pandas as pd

from src.feature_engineering import build_customer_features, default_as_of_from_tickets


def build_features(customers_path, tickets_path, output_path):

    customers = pd.read_csv(customers_path)
    tickets = pd.read_csv(tickets_path)
    as_of = default_as_of_from_tickets(tickets)
    features = build_customer_features(customers, tickets, as_of)
    features.to_csv(output_path, index=False)

    print("Feature dataset saved to:", output_path)
    print("Shape:", features.shape)


if __name__ == "__main__":

    build_features(
        customers_path="../data/processed/customers.csv",
        tickets_path="../data/processed/tickets.csv",
        output_path="../data/processed/customer_features.csv",
    )
    print("Feature Dataset created.")
