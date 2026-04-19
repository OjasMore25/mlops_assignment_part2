## Churn Risk Prediction API

ML-based churn prediction microservice built using FastAPI and a trained **sklearn** `Pipeline` (logistic regression on engineered features).  
Training: `python scripts/train.py` produces `models/churn_pipeline.joblib` and `models/training_metrics.json`.

---

## Base URL

[http://localhost:8000](http://localhost:8000)

**Swagger UI (interactive API documentation):**  
[http://localhost:8000/docs](http://localhost:8000/docs)

**OpenAPI specification:**  
[http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 1. Health Check

### Endpoint

```
GET /
```

### Description

Verifies that the service is running and accessible.

### Response

**200 OK**

```json
{
  "status": "service running"
}
```

---

## 2. Predict Churn Risk

### Endpoint

```http
POST /predict-risk
```

### Description

Computes churn probability and a **LOW / MEDIUM / HIGH** risk band using the same feature engineering as training (ticket frequencies 7d/30d/90d, sentiment score, ticket-type counts, mean days between tickets, charge-change proxy, contract, tenure, charges).

### Request Body

```json
{
  "customer_id": "7590-VHVEG"
}
```

### Successful Response

**200 OK**

```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.12,
  "risk_category": "LOW"
}
```

* `churn_probability`: model-estimated P(churn), between 0 and 1.
* `risk_category`: bands from probability (`< 0.35` → LOW, `< 0.55` → MEDIUM, else HIGH).

### Risk Categories

* `LOW`
* `MEDIUM`
* `HIGH`

---

## 3. Error Responses

### 404 — Customer Not Found

Returned when the provided `customer_id` does not exist.

```json
{
  "detail": "Customer not found"
}
```

---

### 422 — Invalid Request Payload

Returned when the request body:

* Is missing required fields
* Has incorrect field names
* Has incorrect data types

Example invalid request:

```json
{}
```

---

## 4. Metrics Endpoint

### Endpoint

```http
GET /metrics
```

### Description

Exposes application metrics in Prometheus format for monitoring and observability.  
This endpoint is intended for scraping by Prometheus and should not be used directly by clients.

### Example Metrics

* `prediction_count_total`
* `api_request_count_total`
* `api_request_latency_seconds`
* `feature_validation_failures_total` — Pandera schema failures on `/predict-risk`
* `churn_service_resident_memory_bytes` — RSS when `psutil` is installed
* `process_cpu_seconds_total` (Prometheus defaults, if enabled)
* `process_resident_memory_bytes` (Prometheus defaults, if enabled)

---

## 5. Observability Architecture

The service integrates with a monitoring stack:

```text
FastAPI Service
      ↓
/metrics endpoint
      ↓
Prometheus (metrics collection)
      ↓
Grafana (dashboard visualization)
```

Monitoring provides:

* Total prediction count
* Request rate
* API latency (P95)
* System resource usage

---

## 6. Deployment

The service is:

* Containerized using Docker
* Built and pushed automatically via CI/CD
* Published to DockerHub
* Deployable using Docker or Docker Compose

To run locally:

```bash
docker build -t churn-risk-service .
docker run -p 8000:8000 churn-risk-service
```

---

## 7. Version

Current API Version:

```text
2.0
```

```