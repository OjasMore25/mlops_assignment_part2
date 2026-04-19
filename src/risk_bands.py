"""Map churn probability to LOW / MEDIUM / HIGH bands (Assignment II UX)."""


def churn_probability_to_risk_category(p: float) -> str:
    if p < 0.35:
        return "LOW"
    if p < 0.55:
        return "MEDIUM"
    return "HIGH"
