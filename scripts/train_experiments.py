#!/usr/bin/env python3
"""
Train multiple sklearn pipelines, compare in MLflow, save best artifact, promote Production.

Used by DVC `train` stage and Docker/CI for assignment-style model selection.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import default_as_of_from_tickets, feature_columns_for_model  # noqa: E402
from src.model_factory import build_churn_pipeline  # noqa: E402

MODEL_KINDS = ["logistic_regression", "random_forest", "gradient_boosting"]


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


def _score(metrics: dict) -> float:
    return 0.45 * metrics["roc_auc"] + 0.35 * metrics["f1"] + 0.20 * metrics["average_precision"]


def promote_best_model_version(client, registered_name: str, best_run_id: str) -> int | None:
    versions = list(client.search_model_versions(f"name='{registered_name}'"))
    best_version = None
    for mv in versions:
        if mv.run_id == best_run_id:
            best_version = int(mv.version)
            break
    if best_version is None:
        print("No model version matched best run_id; skip registry promotion.")
        return None

    for mv in versions:
        v = int(mv.version)
        if v == best_version:
            continue
        try:
            client.transition_model_version_stage(
                name=registered_name,
                version=v,
                stage="Archived",
                archive_existing_versions=False,
            )
        except Exception as exc:
            print(f"Archive version {v} skipped: {exc}")

    try:
        client.transition_model_version_stage(
            name=registered_name,
            version=best_version,
            stage="Production",
            archive_existing_versions=False,
        )
        print(f"MLflow: model {registered_name} v{best_version} → Production (run {best_run_id})")
    except Exception as exc:
        print(f"Production transition skipped: {exc}")
    return best_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--tickets", type=Path, default=ROOT / "data/processed/tickets.csv")
    parser.add_argument("--out-model", type=Path, default=ROOT / "models/churn_pipeline.joblib")
    parser.add_argument("--out-metrics", type=Path, default=ROOT / "models/training_metrics.json")
    parser.add_argument("--out-manifest", type=Path, default=ROOT / "models/production_manifest.json")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", type=str, default="churn_model_comparison")
    parser.add_argument("--registered-model-name", type=str, default="churn_sklearn")
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument(
        "--model-kinds",
        type=str,
        default=",".join(MODEL_KINDS),
        help="Comma-separated subset of: logistic_regression,random_forest,gradient_boosting",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    tickets = pd.read_csv(args.tickets)
    as_of = default_as_of_from_tickets(tickets)

    y_train = (train_df["Churn"] == "Yes").astype(int)
    y_test = (test_df["Churn"] == "Yes").astype(int)
    cols = feature_columns_for_model(train_df)
    X_train = train_df[cols]
    X_test = test_df[cols]

    kinds = [k.strip() for k in args.model_kinds.split(",") if k.strip()]
    dvc_rev = os.environ.get("DVC_REV") or _try_dvc_rev()

    best = None
    best_run_id: str | None = None
    tracking = args.mlflow_tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}"
    if not args.skip_mlflow:
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(args.experiment_name)

    for model_kind in kinds:
        print(f"--- Training: {model_kind} ---")
        pipeline = build_churn_pipeline(cols, model_kind)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = pipeline.predict(X_test)
        metrics = {
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
        }
        score = _score(metrics)
        print(json.dumps({"model_kind": model_kind, **metrics, "selection_score": score}, indent=2))

        run_id = None
        if not args.skip_mlflow:
            with mlflow.start_run(run_name=model_kind) as run:
                run_id = run.info.run_id
                mlflow.log_params(
                    {
                        "model_kind": model_kind,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "n_features": len(cols),
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.log_metric("selection_score", score)
                mlflow.log_text("\n".join(cols), "feature_list.txt")
                if dvc_rev:
                    mlflow.set_tag("dvc_rev", dvc_rev)
                mlflow.set_tag("dataset_as_of", as_of.isoformat())
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    registered_model_name=args.registered_model_name,
                )

        candidate = {
            "model_kind": model_kind,
            "metrics": metrics,
            "selection_score": score,
            "pipeline": pipeline,
            "run_id": run_id,
        }
        key = (score, metrics["roc_auc"], metrics["f1"])
        if best is None:
            better = True
        else:
            prev = (
                best["selection_score"],
                best["metrics"]["roc_auc"],
                best["metrics"]["f1"],
            )
            better = key > prev
        if better:
            best = candidate
            best_run_id = run_id

    assert best is not None
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], args.out_model)

    out_metrics = {
        **best["metrics"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "as_of": as_of.isoformat(),
        "feature_columns": cols,
        "best_model_kind": best["model_kind"],
        "selection_score": best["selection_score"],
    }
    if dvc_rev:
        out_metrics["dvc_rev"] = dvc_rev
    args.out_metrics.write_text(json.dumps(out_metrics, indent=2))

    manifest = {
        "best_model_kind": best["model_kind"],
        "mlflow_run_id": best_run_id,
        "registered_model_name": args.registered_model_name,
        "selection_score": best["selection_score"],
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2))

    print("Best model:", json.dumps(manifest, indent=2))
    print("Saved:", args.out_model)

    if not args.skip_mlflow and best_run_id:
        from mlflow.tracking import MlflowClient

        tracking = args.mlflow_tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}"
        client = MlflowClient(tracking_uri=tracking)
        promote_best_model_version(client, args.registered_model_name, best_run_id)


if __name__ == "__main__":
    main()
