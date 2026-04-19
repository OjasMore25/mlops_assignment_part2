"""Pandera schema for inference-time feature rows (Part 2 — schema validation)."""

from __future__ import annotations

import pandera as pa


def make_feature_schema(feature_columns: list[str]) -> pa.DataFrameSchema:
    col_specs: dict[str, pa.Column] = {}
    for c in feature_columns:
        if c == "contract_type":
            col_specs[c] = pa.Column(str, nullable=False)
        else:
            col_specs[c] = pa.Column(float, nullable=True, coerce=True)
    return pa.DataFrameSchema(col_specs, strict=True, ordered=False)
