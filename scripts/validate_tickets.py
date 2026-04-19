import pandas as pd

customers = pd.read_csv("../data/processed/customers.csv")
tickets = pd.read_csv("../data/processed/tickets.csv")

print("\nCustomers shape:", customers.shape)
print("Tickets shape:", tickets.shape)

print("\nSample tickets:")
print(tickets.head())

print("\nTicket type distribution:")
print(tickets["ticket_type"].value_counts())

print("\nNull values in tickets:")
print(tickets.isnull().sum())

invalid_ids = tickets[~tickets["customer_id"].isin(customers["customer_id"])]
print("\nInvalid ticket customer IDs:", len(invalid_ids))

ticket_counts = tickets.groupby("customer_id").size().reset_index(name="ticket_count")

merged = customers.merge(ticket_counts, on="customer_id", how="left")
merged["ticket_count"] = merged["ticket_count"].fillna(0)

print("\nTicket count distribution:")
print(merged["ticket_count"].describe())

print("\nTicket count by churn:")
print(merged.groupby("Churn")["ticket_count"].describe())

tickets["created_at"] = pd.to_datetime(tickets["created_at"])

recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)

recent_tickets = tickets[tickets["created_at"] > recent_cutoff]

recent_counts = recent_tickets.groupby("customer_id").size().reset_index(name="recent_tickets")

merged = merged.merge(recent_counts, on="customer_id", how="left")
merged["recent_tickets"] = merged["recent_tickets"].fillna(0)

print("\nRecent tickets by churn:")
print(merged.groupby("Churn")["recent_tickets"].describe())