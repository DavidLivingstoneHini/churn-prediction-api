"""
Admin endpoints — analytics dashboard data, drift reports, model info.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentAdmin
from app.database.models import BatchJob, DriftLog, PredictionLog, PredictionStatus
from app.database.session import get_db
from app.ml.drift import check_drift
from app.ml.features import ALL_FEATURES, NUMERIC_FEATURES, engineer_features
from app.ml.predictor import predictor

MODELS_DIR = Path("/app/models")

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Schemas ───────────────────────────────────────────────────

class ModelInfo(BaseModel):
    version: str
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    training_rows: int
    churn_rate: float
    feature_count: int


class AnalyticsResponse(BaseModel):
    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_churn_probability: float
    avg_inference_time_ms: float
    predictions_today: int
    predictions_this_week: int
    daily_volume: list[dict]
    risk_distribution: list[dict]
    churn_probability_histogram: list[dict]
    top_batch_jobs: list[dict]


class DriftReport(BaseModel):
    drift_detected: bool
    max_psi: float
    avg_psi: float
    drifted_features: dict
    monitored_features: dict
    stable_feature_count: int
    threshold: float
    n_reference: int
    n_current: int
    checked_at: str


# ── Endpoints ─────────────────────────────────────────────────

@router.get("/model", response_model=ModelInfo)
async def get_model_info(current_admin: CurrentAdmin):
    """Return current active model metrics."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found")

    meta = json.loads(meta_path.read_text())
    return ModelInfo(
        version=meta.get("version", "1.0.0"),
        auc_roc=meta.get("auc_roc", 0),
        precision=meta.get("precision", 0),
        recall=meta.get("recall", 0),
        f1_score=meta.get("f1_score", 0),
        training_rows=meta.get("training_rows", 0),
        churn_rate=meta.get("churn_rate", 0),
        feature_count=len(ALL_FEATURES),
    )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    current_admin: CurrentAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start  = now - timedelta(days=7)

    total = (await db.execute(select(func.count(PredictionLog.id)))).scalar() or 0

    high = (await db.execute(
        select(func.count(PredictionLog.id))
        .where(PredictionLog.risk_level == PredictionStatus.HIGH)
    )).scalar() or 0

    medium = (await db.execute(
        select(func.count(PredictionLog.id))
        .where(PredictionLog.risk_level == PredictionStatus.MEDIUM)
    )).scalar() or 0

    low = (await db.execute(
        select(func.count(PredictionLog.id))
        .where(PredictionLog.risk_level == PredictionStatus.LOW)
    )).scalar() or 0

    avg_prob = (await db.execute(
        select(func.avg(PredictionLog.churn_probability))
    )).scalar() or 0.0

    avg_rt = (await db.execute(
        select(func.avg(PredictionLog.inference_time_ms))
        .where(PredictionLog.inference_time_ms.isnot(None))
    )).scalar() or 0.0

    today_count = (await db.execute(
        select(func.count(PredictionLog.id))
        .where(PredictionLog.created_at >= today_start)
    )).scalar() or 0

    week_count = (await db.execute(
        select(func.count(PredictionLog.id))
        .where(PredictionLog.created_at >= week_start)
    )).scalar() or 0

    # Daily volume — last 14 days
    daily_volume = []
    for days_ago in range(13, -1, -1):
        day       = now - timedelta(days=days_ago)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        count = (await db.execute(
            select(func.count(PredictionLog.id))
            .where(PredictionLog.created_at >= day_start,
                   PredictionLog.created_at <  day_end)
        )).scalar() or 0
        daily_volume.append({"date": day_start.strftime("%b %d"), "predictions": count})

    # Risk distribution
    risk_distribution = [
        {"risk": "High",   "count": high},
        {"risk": "Medium", "count": medium},
        {"risk": "Low",    "count": low},
    ]

    # Churn probability histogram (10 buckets 0.0–1.0)
    logs_result = await db.execute(
        select(PredictionLog.churn_probability)
        .order_by(PredictionLog.created_at.desc())
        .limit(5000)
    )
    probs = [r[0] for r in logs_result.all()]

    histogram = []
    for i in range(10):
        lo = i * 0.1
        hi = lo + 0.1
        bucket_count = sum(1 for p in probs if lo <= p < hi)
        histogram.append({
            "range":  f"{int(lo*100)}–{int(hi*100)}%",
            "count":  bucket_count,
        })

    # Recent batch jobs
    batch_result = await db.execute(
        select(BatchJob)
        .order_by(BatchJob.created_at.desc())
        .limit(5)
    )
    top_batch = [
        {
            "id": str(j.id),
            "filename": j.filename,
            "total_rows": j.total_rows,
            "status": j.status,
            "created_at": j.created_at.isoformat(),
            "summary": j.result_summary or {},
        }
        for j in batch_result.scalars().all()
    ]

    return AnalyticsResponse(
        total_predictions=total,
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        avg_churn_probability=round(float(avg_prob), 4),
        avg_inference_time_ms=round(float(avg_rt), 1),
        predictions_today=today_count,
        predictions_this_week=week_count,
        daily_volume=daily_volume,
        risk_distribution=risk_distribution,
        churn_probability_histogram=histogram,
        top_batch_jobs=top_batch,
    )


@router.get("/drift", response_model=DriftReport)
async def check_data_drift(
    current_admin: CurrentAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Compute PSI-based data drift between training distribution
    and recent inference requests (last 500 predictions).
    """
    train_path = MODELS_DIR / "training_data.csv"
    if not train_path.exists():
        raise HTTPException(status_code=404, detail="Training data not found")

    reference_df = pd.read_csv(train_path)
    reference_features = engineer_features(reference_df)

    # Pull recent predictions from DB
    result = await db.execute(
        select(PredictionLog.input_features)
        .order_by(PredictionLog.created_at.desc())
        .limit(500)
    )
    rows = [r[0] for r in result.all()]

    if len(rows) < 30:
        raise HTTPException(
            status_code=400,
            detail="Not enough recent predictions for drift analysis (need ≥30)"
        )

    current_df = pd.DataFrame(rows)
    current_features = engineer_features(current_df)

    drift = check_drift(
        reference_df=pd.DataFrame(reference_features, columns=ALL_FEATURES),
        current_df=pd.DataFrame(current_features,   columns=ALL_FEATURES),
        features=NUMERIC_FEATURES,
        threshold=0.2,
    )

    # Persist drift log
    from app.database.models import DriftLog
    for feat, psi in drift["feature_psi"].items():
        db.add(DriftLog(
            feature_name=feat,
            psi_score=psi,
            drift_detected=psi >= 0.2,
        ))
    await db.commit()

    return DriftReport(
        drift_detected=drift["drift_detected"],
        max_psi=round(drift["max_psi"], 4),
        avg_psi=round(drift["avg_psi"], 4),
        drifted_features=drift["drifted_features"],
        monitored_features=drift["monitored_features"],
        stable_feature_count=len(drift["stable_features"]),
        threshold=drift["threshold"],
        n_reference=drift["n_reference"],
        n_current=drift["n_current"],
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/predictions/history")
async def prediction_history(
    current_admin: CurrentAdmin,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 100,
    offset: int = 0,
):
    """Paginated prediction audit log."""
    result = await db.execute(
        select(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    logs = result.scalars().all()
    total = (await db.execute(select(func.count(PredictionLog.id)))).scalar() or 0

    return {
        "total": total,
        "items": [
            {
                "id":                str(log.id),
                "customer_id":       log.customer_id,
                "churn_probability": log.churn_probability,
                "risk_level":        log.risk_level.value,
                "model_version":     log.model_version,
                "inference_time_ms": log.inference_time_ms,
                "created_at":        log.created_at.isoformat(),
            }
            for log in logs
        ],
    }
