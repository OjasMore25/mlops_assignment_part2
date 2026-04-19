from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_check():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "service running"


def test_predict_risk_valid_customer():

    response = client.post(
        "/predict-risk",
        json={"customer_id": "7590-VHVEG"}
    )

    assert response.status_code == 200
    body = response.json()
    assert "risk_category" in body
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_risk_invalid_customer():

    response = client.post(
        "/predict-risk",
        json={"customer_id": "INVALID-ID"}
    )

    assert response.status_code == 404