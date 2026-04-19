#!/usr/bin/env python3
"""
Train churn classifier (sklearn Pipeline). Logs F1, ROC-AUC, and average precision (PR).
Part 2 PDF — Task 2 + MLOps: MLflow experiment tracking, optional model registry stages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    build_customer_features,
    default_as_of_from_tickets,
    feature_columns_for_model,
)
from src.model_factory import build_churn_pipeline  # noqa: E402


def _try_dvc_rev() -> str | None:
    try:
        out = subprocess.run(
            ["dvc", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=Path, default=ROOT / "data/processed/customers.csv")
    parser.add_argument("--tickets", type=Path, default=ROOT / "data/processed/tickets.csv")
    parser.add_argument("--train-csv", type=Path, default=None, help="If set with --test-csv, train only on this split")
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--out-model", type=Path, default=ROOT / "models/churn_pipeline.joblib")
    parser.add_argument("--out-metrics", type=Path, default=ROOT / "models/training_metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", type=str, default="churn_prediction")
    parser.add_argument("--registered-model-name", type=str, default="churn_sklearn")
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument(
        "--model-kind",
        type=str,
        default="logistic_regression",
        help="logistic_regression | random_forest | gradient_boosting",
    )
    args = parser.parse_args()

    if args.train_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        as_of = default_as_of_from_tickets(pd.read_csv(args.tickets))
        y_train = (train_df["Churn"] == "Yes").astype(int)
        y_test = (test_df["Churn"] == "Yes").astype(int)
        cols = feature_columns_for_model(train_df)
        X_train = train_df[cols]
        X_test = test_df[cols]
    else:
        customers = pd.read_csv(args.customers)
        tickets = pd.read_csv(args.tickets)
        as_of = default_as_of_from_tickets(tickets)
        features = build_customer_features(customers, tickets, as_of)
        y = (features["Churn"] == "Yes").astype(int)
        cols = feature_columns_for_model(features)
        X = features[cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
        )

    model = build_churn_pipeline(cols, args.model_kind)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    metrics = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "as_of": as_of.isoformat(),
        "feature_columns": cols,
        "model_kind": args.model_kind,
    }

    dvc_rev = os.environ.get("DVC_REV") or _try_dvc_rev()
    if dvc_rev:
        metrics["dvc_rev"] = dvc_rev

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)
    args.out_metrics.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print("Saved model:", args.out_model)

    if not args.skip_mlflow:
        tracking = args.mlflow_tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}"
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "random_state": args.random_state,
                    "test_size": args.test_size,
                    "n_features": len(cols),
                    "split_mode": "files" if args.train_csv else "random",
                    "model_kind": args.model_kind,
                }
            )
            mlflow.log_metrics(
                {
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                }
            )
            mlflow.log_text("\n".join(cols), "feature_list.txt")
            if dvc_rev:
                mlflow.set_tag("dvc_rev", dvc_rev)
            mlflow.set_tag("dataset_as_of", metrics["as_of"])

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=args.registered_model_name,
            )

        # Stage transition: Staging for latest version (PDF — model registry)
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=tracking)
            versions = client.search_model_versions(f"name='{args.registered_model_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name=args.registered_model_name,
                    version=latest.version,
                    stage="Staging",
                    archive_existing_versions=False,
                )
                print(f"MLflow: transitioned model version {latest.version} to Staging")
        except Exception as e:
            print("MLflow registry stage transition skipped:", e)


if __name__ == "__main__":
    main()
