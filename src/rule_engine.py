import pandas as pd


def compute_risk(row):

    tickets_30 = row["tickets_last_30_days"]
    contract = row["contract_type"]
    complaint = row["complaint_ticket"]

    # Rule 1
    if tickets_30 > 5:
        return "HIGH"

    # Rule 2
    if contract == "Month-to-month" and complaint == 1:
        return "HIGH"

    # Rule 3
    if tickets_30 >= 3:
        return "MEDIUM"

    return "LOW"


def apply_rules(input_path, output_path):

    df = pd.read_csv(input_path)

    df["risk_category"] = df.apply(compute_risk, axis=1)

    df.to_csv(output_path, index=False)

    print("Risk predictions generated")
    print(df["risk_category"].value_counts())


if __name__ == "__main__":

    apply_rules(
        "../data/processed/customer_features.csv",
        "../data/processed/customer_risk_predictions.csv"
    )