import pandas as pd
import pandera.errors
import pytest

from src.inference_schema import make_feature_schema


def test_schema_accepts_valid_row():
    cols = ["contract_type", "monthly_charges", "tenure"]
    schema = make_feature_schema(cols)
    df = pd.DataFrame(
        [{"contract_type": "One year", "monthly_charges": 50.0, "tenure": 12.0}]
    )
    schema.validate(df)


def test_schema_rejects_bad_contract_type_dtype():
    cols = ["contract_type", "monthly_charges"]
    schema = make_feature_schema(cols)
    df = pd.DataFrame([{"contract_type": 123, "monthly_charges": 1.0}])
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(df)
