# Churn Risk Prediction Service (DevOps + MLOps Architecture)

This project implements a **churn risk prediction microservice** with **FastAPI**, a full **DevOps lifecycle** (containerization, CI/CD, monitoring), and an **MLOps Part 2** track: versioned data (DVC), multi-model training with **MLflow**, drift checks tied to the **ticket generator**, optional **CT** workflows, and **sklearn `Pipeline`** inference.

The service predicts churn risk from **customer attributes** and **support ticket behaviour** (Telco-style data plus synthetic tickets).

The project demonstrates two complementary architectures:

- **DevOps Architecture** — infrastructure, CI/CD, containerization, monitoring
- **MLOps Architecture** — feature engineering, experiment tracking, model registry, drift-aware retraining hooks, inference API

---

## Train and reproduce (MLOps Part 2)

**Recommended — DVC pipeline (splits + multi-model train):**

```bash
dvc repro
```

This runs `split` → `train` per [`dvc.yaml`](dvc.yaml) and writes `data/splits/*.csv`, `models/churn_pipeline.joblib`, and `models/training_metrics.json` (large artifacts are gitignored; CI/Docker regenerate them).

Training logs to **MLflow** (SQLite `mlflow.db` in repo root; gitignored) and registers **`churn_sklearn`**, promoting the best run to **Production**.

**MLflow UI** (paths with spaces must quote the URI):

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --host 127.0.0.1 --port 5000
```

**Single-model ad-hoc train:**

```bash
python scripts/train.py --model-kind logistic_regression
```

**Local tests:** [`conftest.py`](conftest.py) creates splits + a model if missing (`pytest`, with `--skip-mlflow` in the bootstrap path).

**Quality / drift:**

```bash
python scripts/metric_gate.py
python scripts/check_drift.py
python scripts/simulate_ticket_drift.py   # writes tickets_drifted.csv for drift demos
```

**Full stage-wise guide (screenshots, drift → CT):**  
[`docs/MLOps_Assignment2_EndToEnd.md`](docs/MLOps_Assignment2_EndToEnd.md)

---

# Project objectives

Demonstrate a production-oriented ML service with modern DevOps and MLOps practices: REST inference, automated tests, Docker images, GitHub Actions, Prometheus metrics, and experiment/model lifecycle tooling.

---

# Repository structure (high level)

```
MLOps-A01
├── src/                 # app, feature_engineering, model_factory, ticket generator, …
├── data/processed/      # customers, tickets, customer_features (CSV)
├── data/splits/         # DVC-managed train/val/test (see data/splits/.gitignore)
├── scripts/             # train_experiments, drift, DVC helpers, data scripts
├── tests/
├── monitoring/
├── docs/                # API, walkthrough, screenshots, execution log
├── .github/workflows/
├── Dockerfile
├── dvc.yaml
└── requirements.txt
```

---

# DevOps architecture

Pipeline: GitHub → CI (tests, DVC repro, metric gate) → Docker build/push → deployment targets.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and the monitoring stack under [`monitoring/`](monitoring/).

---

# MLOps architecture (implemented)

| Stage | Implementation |
|-------|----------------|
| Data / splits | DVC [`dvc.yaml`](dvc.yaml), [`scripts/create_splits.py`](scripts/create_splits.py) |
| Features | [`src/feature_engineering.py`](src/feature_engineering.py) |
| Training | [`scripts/train_experiments.py`](scripts/train_experiments.py) (Logistic / RF / GBDT) |
| Tracking | MLflow experiment `churn_model_comparison`, registry `churn_sklearn` |
| Inference | [`src/app.py`](src/app.py) + Pandera [`src/inference_schema.py`](src/inference_schema.py) |
| Drift / CT | [`scripts/check_drift.py`](scripts/check_drift.py), [`.github/workflows/ct_on_drift.yml`](.github/workflows/ct_on_drift.yml) |

---

# Running the service

### Locally

```bash
source .venv/bin/activate   # after pip install -r requirements.txt
dvc repro                   # or ensure models/churn_pipeline.joblib exists
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker

The image runs **split + `train_experiments --skip-mlflow`** during build, then starts the API.

```bash
docker build -t churn-risk-service .
docker run --rm -p 8000:8000 churn-risk-service
```

`POST /predict-risk` returns `churn_probability` and `risk_category` (LOW / MEDIUM / HIGH). See [`docs/API.md`](docs/API.md).

---

# Monitoring

FastAPI exposes `/metrics` for Prometheus; optional Grafana compose under [`monitoring/`](monitoring/).

---

# Testing

```bash
pytest -q
```

Sixteen tests cover the API, feature engineering, schema validation, model factory, drift detection, legacy rule engine, and training smoke paths. Inventory: [`docs/MLOps_Assignment2_EndToEnd.md`](docs/MLOps_Assignment2_EndToEnd.md) (Stage 8).

---

# Dataset

Telco-style **customers** plus **synthetic tickets** from [`src/ticket_generator.py`](src/ticket_generator.py) / [`scripts/generate_tickets.py`](scripts/generate_tickets.py).

---

# Technologies

| Area | Stack |
|------|--------|
| API | FastAPI, Uvicorn |
| ML | scikit-learn, joblib, MLflow |
| Data ops | DVC |
| Quality | pytest, httpx, pandera |
| Ops | Docker, GitHub Actions, Prometheus |

---

# Ojas - 2022BCD0043
