"""
Tests for the feature engineering pipeline.
All tests are pure Python — no DB, no model required.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    engineer_single,
)


# ── Helpers ───────────────────────────────────────────────────

def make_row(**overrides) -> dict:
    base = {
        "tenure":            12,
        "MonthlyCharges":    65.00,
        "TotalCharges":      780.00,
        "SeniorCitizen":     0,
        "Partner":           "No",
        "Dependents":        "No",
        "PhoneService":      "Yes",
        "MultipleLines":     "No",
        "InternetService":   "Fiber optic",
        "OnlineSecurity":    "No",
        "OnlineBackup":      "No",
        "DeviceProtection":  "No",
        "TechSupport":       "No",
        "StreamingTV":       "No",
        "StreamingMovies":   "No",
        "Contract":          "Month-to-month",
        "PaperlessBilling":  "Yes",
        "PaymentMethod":     "Electronic check",
    }
    base.update(overrides)
    return base


def make_df(rows=5, **overrides) -> pd.DataFrame:
    return pd.DataFrame([make_row(**overrides)] * rows)


# ── Feature count ─────────────────────────────────────────────

def test_feature_count():
    """ALL_FEATURES = 7 numeric + 12 binary + 10 one-hot categorical = 29 total."""
    assert len(ALL_FEATURES) == 29
    assert len(NUMERIC_FEATURES) == 7
    assert len(BINARY_FEATURES) == 12
    assert len(CATEGORICAL_FEATURES) == 10


def test_feature_names_unique():
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))


# ── engineer_features shape ───────────────────────────────────

def test_basic_feature_engineering():
    df = make_df(rows=3)
    out = engineer_features(df)
    assert out.shape == (3, 29)
    assert list(out.columns) == ALL_FEATURES


def test_output_is_float():
    df = make_df(rows=2)
    out = engineer_features(df)
    assert out.dtypes.apply(lambda t: t == float).all(), "All output features must be float"


def test_no_nan_in_output():
    df = make_df(rows=10)
    out = engineer_features(df)
    assert not out.isnull().any().any(), "Feature matrix must not contain NaN"


# ── Engineered numeric features ───────────────────────────────

def test_engineered_features():
    df = make_df(
        rows=1,
        tenure=12,
        MonthlyCharges=60.0,
        TotalCharges=720.0,
    )
    out = engineer_features(df)

    # avg_monthly_spend = total / tenure
    assert abs(out["avg_monthly_spend"].iloc[0] - 60.0) < 0.5

    # tenure_monthly_ratio = tenure / monthly_charges
    assert abs(out["tenure_monthly_ratio"].iloc[0] - (12 / 60.0)) < 0.1


def test_zero_tenure_no_crash():
    """Edge case: tenure=0 should not produce NaN or crash."""
    df = make_df(rows=1, tenure=0, TotalCharges=0.0)
    out = engineer_features(df)
    assert not out.isnull().any().any()


def test_high_service_count():
    """Customer with all services should have higher num_services."""
    all_services = dict(
        PhoneService="Yes", MultipleLines="Yes",
        OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes",
        StreamingTV="Yes", StreamingMovies="Yes",
    )
    df_high = make_df(rows=1, **all_services)
    df_low  = make_df(rows=1, PhoneService="No", MultipleLines="No")
    out_high = engineer_features(df_high)
    out_low  = engineer_features(df_low)
    assert out_high["num_services"].iloc[0] > out_low["num_services"].iloc[0]


# ── Categorical one-hot ───────────────────────────────────────

def test_internet_service_ohe_fiber():
    df = make_df(rows=1, InternetService="Fiber optic")
    out = engineer_features(df)
    assert out["internet_service_fiber"].iloc[0] == 1
    assert out["internet_service_dsl"].iloc[0] == 0
    assert out["internet_service_no"].iloc[0] == 0


def test_internet_service_ohe_dsl():
    df = make_df(rows=1, InternetService="DSL")
    out = engineer_features(df)
    assert out["internet_service_dsl"].iloc[0] == 1
    assert out["internet_service_fiber"].iloc[0] == 0


def test_contract_ohe():
    for contract, expected_col in [
        ("Month-to-month", "contract_month"),
        ("One year",       "contract_one_year"),
        ("Two year",       "contract_two_year"),
    ]:
        df = make_df(rows=1, Contract=contract)
        out = engineer_features(df)
        assert out[expected_col].iloc[0] == 1, f"Expected {expected_col}=1 for {contract}"


def test_payment_method_ohe():
    for method, expected_col in [
        ("Electronic check",          "payment_electronic_check"),
        ("Mailed check",              "payment_mailed_check"),
        ("Bank transfer (automatic)", "payment_bank_transfer"),
        ("Credit card (automatic)",   "payment_credit_card"),
    ]:
        df = make_df(rows=1, PaymentMethod=method)
        out = engineer_features(df)
        assert out[expected_col].iloc[0] == 1, f"Expected {expected_col}=1 for {method}"


# ── Binary features ───────────────────────────────────────────

def test_binary_yes_no():
    df_yes = make_df(rows=1, Partner="Yes")
    df_no  = make_df(rows=1, Partner="No")
    out_yes = engineer_features(df_yes)
    out_no  = engineer_features(df_no)
    assert out_yes["partner"].iloc[0] == 1
    assert out_no["partner"].iloc[0]  == 0


def test_senior_citizen_int():
    df = make_df(rows=1, SeniorCitizen=1)
    out = engineer_features(df)
    assert out["senior_citizen"].iloc[0] == 1


# ── engineer_single ───────────────────────────────────────────

def test_single_inference_shape():
    row = make_row()
    result = engineer_single(row)
    assert result.shape == (1, 29)


def test_single_inference_no_nan():
    row = make_row()
    result = engineer_single(row)
    assert not np.isnan(result).any()


def test_single_inference_dtype():
    row = make_row()
    result = engineer_single(row)
    assert result.dtype == np.float64 or result.dtype == np.float32 or "float" in str(result.dtype)
