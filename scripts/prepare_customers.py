import pandas as pd

df = pd.read_csv("../data/raw/telco-churn.csv")

# fix TotalCharges issue
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # sometimes contains blanks

df = df.dropna(subset=["TotalCharges"])

# rename important columns
df = df.rename(columns={
    "customerID": "customer_id",
    "Contract": "contract_type",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges"
})

# keep useful columns
columns_to_keep = [
    "customer_id",
    "contract_type",
    "tenure",
    "monthly_charges",
    "total_charges",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen",
    "Churn"
]

df = df[columns_to_keep]

df.to_csv("../data/processed/customers.csv", index=False)

print("customers dataset saved")