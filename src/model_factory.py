"""Sklearn Pipelines for churn experiments (multiple estimators, shared preprocessing)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess(feature_names: list[str]) -> ColumnTransformer:
    cat_cols = ["contract_type"]
    num_cols = [c for c in feature_names if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def build_churn_pipeline(feature_names: list[str], model_kind: str) -> Pipeline:
    prep = build_preprocess(feature_names)
    kind = model_kind.lower().strip()

    if kind == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif kind == "gradient_boosting":
        clf = GradientBoostingClassifier(
            random_state=42,
            learning_rate=0.08,
            n_estimators=150,
            subsample=0.9,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    return Pipeline([("prep", prep), ("clf", clf)])
