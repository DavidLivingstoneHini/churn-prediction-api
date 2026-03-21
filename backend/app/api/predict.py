"""
Prediction endpoints — single customer and batch CSV upload.
"""
from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.database.models import BatchJob, PredictionLog, PredictionStatus
from app.database.session import get_db
from app.ml.features import ALL_FEATURES
from app.ml.predictor import predictor

router = APIRouter(prefix="/predict", tags=["predict"])


# ── Request / response schemas ─────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw customer features — matches Telco Churn dataset fields."""
    tenure:            int   = Field(..., ge=0, le=120,  description="Months as customer")
    monthly_charges:   float = Field(..., ge=0, le=500,  description="Monthly bill amount")
    total_charges:     float = Field(..., ge=0,           description="Total billed to date")
    senior_citizen:    int   = Field(0,  ge=0, le=1)
    partner:           str   = Field("No")
    dependents:        str   = Field("No")
    phone_service:     str   = Field("Yes")
    multiple_lines:    str   = Field("No")
    internet_service:  str   = Field("Fiber optic",       description="DSL | Fiber optic | No")
    online_security:   str   = Field("No")
    online_backup:     str   = Field("No")
    device_protection: str   = Field("No")
    tech_support:      str   = Field("No")
    streaming_tv:      str   = Field("No")
    streaming_movies:  str   = Field("No")
    contract:          str   = Field("Month-to-month",    description="Month-to-month | One year | Two year")
    paperless_billing: str   = Field("Yes")
    payment_method:    str   = Field("Electronic check")
    customer_id:       Optional[str] = Field(None)

    @field_validator("internet_service")
    @classmethod
    def valid_internet(cls, v: str) -> str:
        valid = {"dsl", "fiber optic", "no"}
        if v.lower() not in valid:
            raise ValueError(f"internet_service must be one of: {valid}")
        return v

    @field_validator("contract")
    @classmethod
    def valid_contract(cls, v: str) -> str:
        valid = {"month-to-month", "one year", "two year"}
        if v.lower() not in valid:
            raise ValueError(f"contract must be one of: {valid}")
        return v

    def to_feature_dict(self) -> dict:
        return {
            "tenure":            self.tenure,
            "MonthlyCharges":    self.monthly_charges,
            "TotalCharges":      self.total_charges,
            "SeniorCitizen":     self.senior_citizen,
            "Partner":           self.partner,
            "Dependents":        self.dependents,
            "PhoneService":      self.phone_service,
            "MultipleLines":     self.multiple_lines,
            "InternetService":   self.internet_service,
            "OnlineSecurity":    self.online_security,
            "OnlineBackup":      self.online_backup,
            "DeviceProtection":  self.device_protection,
            "TechSupport":       self.tech_support,
            "StreamingTV":       self.streaming_tv,
            "StreamingMovies":   self.streaming_movies,
            "Contract":          self.contract,
            "PaperlessBilling":  self.paperless_billing,
            "PaymentMethod":     self.payment_method,
        }


class SimilarCustomer(BaseModel):
    similarity_score: float
    churn_label: int
    distance: float


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    customer_id: Optional[str]
    churn_probability: float
    risk_level: str
    model_version: str
    inference_time_ms: int
    similar_customers: list[SimilarCustomer]
    recommendation: str
    created_at: str


class BatchSummary(BaseModel):
    job_id: uuid.UUID
    filename: str
    total_rows: int
    high_risk: int
    medium_risk: int
    low_risk: int
    avg_churn_probability: float
    status: str
    created_at: str


# ── Helpers ───────────────────────────────────────────────────

def _recommendation(prob: float, risk: str) -> str:
    if risk == "high":
        return (
            "Immediate retention action recommended. "
            "Consider offering a discounted long-term contract, "
            "dedicated support, or a loyalty reward."
        )
    if risk == "medium":
        return (
            "Monitor closely. Proactively reach out with personalised offers "
            "or service upgrades to reduce churn risk."
        )
    return "Customer appears stable. Continue standard engagement programme."


def _risk_enum(risk: str) -> PredictionStatus:
    return {
        "low":    PredictionStatus.LOW,
        "medium": PredictionStatus.MEDIUM,
        "high":   PredictionStatus.HIGH,
    }[risk]


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/single", response_model=PredictionResponse, status_code=201)
async def predict_single(
    payload: CustomerFeatures,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Predict churn probability for a single customer.
    Returns calibrated probability, risk tier, FAISS-based
    similar customer profiles, and a recommended action.
    """
    try:
        result = predictor.predict(payload.to_feature_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    log = PredictionLog(
        user_id=current_user.id,
        customer_id=payload.customer_id,
        input_features=payload.to_feature_dict(),
        churn_probability=result.churn_probability,
        risk_level=_risk_enum(result.risk_level),
        model_version=result.model_version,
        inference_time_ms=result.inference_time_ms,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    return PredictionResponse(
        prediction_id=log.id,
        customer_id=payload.customer_id,
        churn_probability=result.churn_probability,
        risk_level=result.risk_level,
        model_version=result.model_version,
        inference_time_ms=result.inference_time_ms,
        similar_customers=[SimilarCustomer(**s) for s in result.similar_customers],
        recommendation=_recommendation(result.churn_probability, result.risk_level),
        created_at=log.created_at.isoformat(),
    )


@router.post("/batch", response_model=BatchSummary, status_code=201)
async def predict_batch(
    file: Annotated[UploadFile, File(...)],
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Batch predict from a CSV upload.
    Required columns: tenure, MonthlyCharges, TotalCharges, Contract,
    InternetService, PaymentMethod + optional demographic fields.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    required = {"tenure", "MonthlyCharges", "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}",
        )

    # Create batch job record
    job = BatchJob(
        user_id=current_user.id,
        filename=file.filename,
        total_rows=len(df),
        status="processing",
    )
    db.add(job)
    await db.flush()

    # Run predictions
    records = df.to_dict(orient="records")
    results = predictor.predict_batch(records)

    risk_counts = {"high": 0, "medium": 0, "low": 0}
    probs = []

    for row, res in zip(records, results):
        risk_counts[res.risk_level] += 1
        probs.append(res.churn_probability)

        db.add(PredictionLog(
            user_id=current_user.id,
            customer_id=str(row.get("customerID", "")),
            input_features=row,
            churn_probability=res.churn_probability,
            risk_level=_risk_enum(res.risk_level),
            model_version=res.model_version,
            inference_time_ms=res.inference_time_ms,
        ))

    summary = {
        "high_risk":   risk_counts["high"],
        "medium_risk": risk_counts["medium"],
        "low_risk":    risk_counts["low"],
        "avg_churn_probability": round(sum(probs) / len(probs), 4) if probs else 0.0,
    }

    job.processed_rows = len(results)
    job.status = "completed"
    job.result_summary = summary
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(job)

    return BatchSummary(
        job_id=job.id,
        filename=file.filename,
        total_rows=job.total_rows,
        status=job.status,
        created_at=job.created_at.isoformat(),
        **summary,
    )
