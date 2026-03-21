"""
Feature engineering for the Telco Customer Churn dataset.
All transformations are deterministic and applied consistently
at training time and inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Canonical feature list (order matters for model) ──────────
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_services",
    "avg_monthly_spend",
    "charge_per_service",
    "tenure_monthly_ratio",
]

BINARY_FEATURES = [
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
]

CATEGORICAL_FEATURES = [
    "internet_service_dsl",
    "internet_service_fiber",
    "internet_service_no",
    "contract_month",
    "contract_one_year",
    "contract_two_year",
    "payment_bank_transfer",
    "payment_credit_card",
    "payment_electronic_check",
    "payment_mailed_check",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def _safe_bool(val) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(bool(val))
    if isinstance(val, str):
        return 1 if val.strip().lower() in ("yes", "true", "1") else 0
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw input dataframe into model-ready feature matrix.
    Handles both training data (from CSV) and inference data (from API).
    """
    out = pd.DataFrame(index=df.index)

    # ── Numeric ───────────────────────────────────────────────
    out["tenure"] = pd.to_numeric(df.get("tenure", 0), errors="coerce").fillna(0)
    out["monthly_charges"] = pd.to_numeric(
        df.get("MonthlyCharges", df.get("monthly_charges", 0)), errors="coerce"
    ).fillna(0)
    out["total_charges"] = pd.to_numeric(
        df.get("TotalCharges", df.get("total_charges", 0)), errors="coerce"
    ).fillna(0)

    # Engineered features — these are the ones interviewers ask about
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    service_count = sum(
        (df.get(c, pd.Series(["No"] * len(df))).str.lower() == "yes").astype(int)
        for c in service_cols
        if c in df.columns
    )
    out["num_services"] = service_count if not isinstance(service_count, int) else 0

    out["avg_monthly_spend"] = np.where(
        out["tenure"] > 0, out["total_charges"] / out["tenure"], out["monthly_charges"]
    )
    out["charge_per_service"] = np.where(
        out["num_services"] > 0,
        out["monthly_charges"] / (out["num_services"] + 1),
        out["monthly_charges"],
    )
    out["tenure_monthly_ratio"] = np.where(
        out["monthly_charges"] > 0,
        out["tenure"] / out["monthly_charges"],
        0,
    )

    # ── Binary ────────────────────────────────────────────────
    bool_map = {
        "senior_citizen":    df.get("SeniorCitizen", df.get("senior_citizen", 0)),
        "partner":           df.get("Partner", df.get("partner", "No")),
        "dependents":        df.get("Dependents", df.get("dependents", "No")),
        "phone_service":     df.get("PhoneService", df.get("phone_service", "No")),
        "multiple_lines":    df.get("MultipleLines", df.get("multiple_lines", "No")),
        "online_security":   df.get("OnlineSecurity", df.get("online_security", "No")),
        "online_backup":     df.get("OnlineBackup", df.get("online_backup", "No")),
        "device_protection": df.get("DeviceProtection", df.get("device_protection", "No")),
        "tech_support":      df.get("TechSupport", df.get("tech_support", "No")),
        "streaming_tv":      df.get("StreamingTV", df.get("streaming_tv", "No")),
        "streaming_movies":  df.get("StreamingMovies", df.get("streaming_movies", "No")),
        "paperless_billing": df.get("PaperlessBilling", df.get("paperless_billing", "No")),
    }
    for col, series in bool_map.items():
        if isinstance(series, pd.Series):
            out[col] = series.apply(_safe_bool)
        else:
            out[col] = _safe_bool(series)

    # ── Categorical (one-hot) ─────────────────────────────────
    internet = (
        df.get("InternetService", df.get("internet_service", "No"))
        .str.lower()
        .str.strip()
        if hasattr(df.get("InternetService", df.get("internet_service", "No")), "str")
        else pd.Series(["no"] * len(df))
    )
    out["internet_service_dsl"]   = (internet == "dsl").astype(int)
    out["internet_service_fiber"] = (internet == "fiber optic").astype(int)
    out["internet_service_no"]    = (internet == "no").astype(int)

    contract = (
        df.get("Contract", df.get("contract", "Month-to-month"))
        .str.lower()
        .str.strip()
        if hasattr(df.get("Contract", df.get("contract", "Month-to-month")), "str")
        else pd.Series(["month-to-month"] * len(df))
    )
    out["contract_month"]    = (contract == "month-to-month").astype(int)
    out["contract_one_year"] = (contract == "one year").astype(int)
    out["contract_two_year"] = (contract == "two year").astype(int)

    payment = (
        df.get("PaymentMethod", df.get("payment_method", "Electronic check"))
        .str.lower()
        .str.strip()
        if hasattr(df.get("PaymentMethod", df.get("payment_method", "Electronic check")), "str")
        else pd.Series(["electronic check"] * len(df))
    )
    out["payment_bank_transfer"]    = (payment == "bank transfer (automatic)").astype(int)
    out["payment_credit_card"]      = (payment == "credit card (automatic)").astype(int)
    out["payment_electronic_check"] = (payment == "electronic check").astype(int)
    out["payment_mailed_check"]     = (payment == "mailed check").astype(int)

    return out[ALL_FEATURES].astype(float)


def engineer_single(data: dict) -> np.ndarray:
    """Engineer features for a single inference request dict."""
    df = pd.DataFrame([data])
    return engineer_features(df).values
