import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import pandera.errors
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from src.feature_engineering import build_customer_features, default_as_of_from_tickets
from src.inference_schema import make_feature_schema
from src.risk_bands import churn_probability_to_risk_category

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("churn-risk-service")

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API request count",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total number of churn predictions made",
)

FEATURE_VALIDATION_FAILURES = Counter(
    "feature_validation_failures_total",
    "Pandera schema validation failures on inference feature rows",
)

SERVICE_RESIDENT_MEMORY_BYTES = Gauge(
    "churn_service_resident_memory_bytes",
    "Approximate RSS of the API process (psutil; distinct from process_* defaults)",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "models/churn_pipeline.joblib"))
METRICS_PATH = Path(os.environ.get("TRAINING_METRICS_PATH", ROOT / "models/training_metrics.json"))
MANIFEST_PATH = Path(os.environ.get("PRODUCTION_MANIFEST_PATH", ROOT / "models/production_manifest.json"))

app = FastAPI(
    title="Churn Risk Prediction Service",
    description="ML-based churn risk prediction API (sklearn pipeline) using customer and ticket data",
    version="2.0",
)

logger.info("Loading datasets...")
customers = pd.read_csv(ROOT / "data/processed/customers.csv")
tickets = pd.read_csv(ROOT / "data/processed/tickets.csv")
tickets["created_at"] = pd.to_datetime(tickets["created_at"])
AS_OF = default_as_of_from_tickets(tickets)

logger.info(
    f"Datasets loaded | customers={len(customers)} | tickets={len(tickets)} | as_of={AS_OF.isoformat()}"
)

if not MODEL_PATH.is_file():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run: python scripts/train.py"
    )

pipeline = joblib.load(MODEL_PATH)
if METRICS_PATH.is_file():
    meta = json.loads(METRICS_PATH.read_text())
    FEATURE_COLUMNS = meta["feature_columns"]
else:
    FEATURE_COLUMNS = list(getattr(pipeline, "feature_names_in_", []))
    if not FEATURE_COLUMNS:
        raise RuntimeError("Could not determine feature columns; provide models/training_metrics.json")

logger.info("Loaded churn pipeline from %s", MODEL_PATH)
if MANIFEST_PATH.is_file():
    try:
        mf = json.loads(MANIFEST_PATH.read_text())
        logger.info("Production manifest: %s", mf)
    except json.JSONDecodeError:
        pass

FEATURE_SCHEMA = make_feature_schema(FEATURE_COLUMNS)


class CustomerRequest(BaseModel):
    customer_id: str


@app.get(
    "/",
    summary="Health Check",
    description="Returns service status to verify that the churn risk service is running.",
    response_description="Service status message",
)
def health_check():

    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    logger.info("Health check endpoint accessed")
    return {"status": "service running"}


@app.get("/metrics")
def metrics():

    logger.info("Metrics endpoint accessed")
    return Response(generate_latest(), media_type="text/plain")


@app.post(
    "/predict-risk",
    summary="Predict Churn Risk",
    description="Churn probability and risk band from the trained sklearn pipeline.",
    response_description="Predicted churn risk",
    responses={
        200: {"description": "Prediction successful"},
        404: {"description": "Customer not found"},
        422: {"description": "Invalid request payload"},
    },
)
def predict_risk(request: CustomerRequest):

    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict-risk").inc()
    customer_id = request.customer_id
    logger.info(f"Prediction request received for customer {customer_id}")

    row = customers[customers["customer_id"] == customer_id]
    if row.empty:
        logger.error(f"Customer not found: {customer_id}")
        raise HTTPException(status_code=404, detail="Customer not found")

    feat_df = build_customer_features(row, tickets, AS_OF)
    X = feat_df[FEATURE_COLUMNS]
    try:
        FEATURE_SCHEMA.validate(X)
    except pandera.errors.SchemaError as exc:
        FEATURE_VALIDATION_FAILURES.inc()
        logger.warning("Feature validation failed: %s", exc)
        raise HTTPException(status_code=422, detail="Feature validation failed") from exc

    if _PSUTIL:
        SERVICE_RESIDENT_MEMORY_BYTES.set(psutil.Process(os.getpid()).memory_info().rss)

    churn_proba = float(pipeline.predict_proba(X)[0, 1])
    risk = churn_probability_to_risk_category(churn_proba)

    PREDICTION_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    logger.info(
        f"Prediction completed | customer={customer_id} | p_churn={churn_proba:.4f} | band={risk}"
    )

    return {
        "customer_id": customer_id,
        "churn_probability": churn_proba,
        "risk_category": risk,
    }
