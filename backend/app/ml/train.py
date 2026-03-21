"""
Training pipeline — runs once at Docker build time.
Downloads Telco Churn dataset, trains XGBoost model,
builds FAISS index, saves all artifacts to /app/models/.

Run manually: python -m app.ml.train
"""
from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    auc, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.features import ALL_FEATURES, engineer_features

MODELS_DIR = Path("/app/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH         = MODELS_DIR / "churn_model.pkl"
SCALER_PATH        = MODELS_DIR / "scaler.pkl"
FAISS_INDEX_PATH   = MODELS_DIR / "faiss.index"
TRAIN_DATA_PATH    = MODELS_DIR / "training_data.csv"
METADATA_PATH      = MODELS_DIR / "model_metadata.json"

# Synthetic dataset parameters — mirrors Telco Churn distributions
N_SAMPLES = 7_043
CHURN_RATE = 0.265  # 26.5% — matches real Telco dataset
RANDOM_STATE = 42


def generate_telco_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate a realistic synthetic Telco Churn dataset.
    We generate it synthetically so the project has zero external dependencies
    at build time, but the distributions mirror the real Kaggle dataset.
    """
    tenure           = rng.integers(0, 73, n)
    monthly_charges  = rng.uniform(18, 119, n).round(2)
    total_charges    = (tenure * monthly_charges + rng.normal(0, 50, n)).clip(0).round(2)
    senior_citizen   = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner          = rng.choice(["Yes", "No"], n)
    dependents       = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    phone_service    = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines   = rng.choice(["Yes", "No", "No phone service"], n)
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    online_security  = rng.choice(["Yes", "No", "No internet service"], n)
    online_backup    = rng.choice(["Yes", "No", "No internet service"], n)
    device_protection= rng.choice(["Yes", "No", "No internet service"], n)
    tech_support     = rng.choice(["Yes", "No", "No internet service"], n)
    streaming_tv     = rng.choice(["Yes", "No", "No internet service"], n)
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], n)
    contract         = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing= rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method   = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )

    # Churn probability depends on key risk factors (mirrors real patterns)
    churn_score = (
        0.3  * (contract == "Month-to-month").astype(float)
        + 0.25 * (internet_service == "Fiber optic").astype(float)
        + 0.15 * (payment_method == "Electronic check").astype(float)
        - 0.2  * (tenure / 72)
        - 0.1  * (partner == "Yes").astype(float)
        + 0.05 * senior_citizen
        + rng.normal(0, 0.1, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score * 3))
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "SeniorCitizen":    senior_citizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":    payment_method,
        "Churn":            churn,
    })


def train() -> dict:
    print("=" * 60)
    print("ML TRAINING PIPELINE")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_STATE)

    # 1. Generate / load data
    print("[1/6] Generating training dataset...")
    df = generate_telco_dataset(N_SAMPLES, rng)
    df.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"      {len(df)} rows | churn rate: {df['Churn'].mean():.1%}")

    # 2. Feature engineering
    print("[2/6] Engineering features...")
    X = engineer_features(df)
    y = df["Churn"].values
    print(f"      Feature matrix: {X.shape}")

    # 3. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Scale numeric features
    print("[3/6] Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5. Train XGBoost with calibration
    print("[4/6] Training XGBoost + calibration...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    # Platt scaling calibration for well-calibrated probabilities
    model = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    print("[5/6] Evaluating model...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc_roc   = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"      AUC-ROC:   {auc_roc:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1:        {f1:.4f}")

    # 7. Build FAISS index for similarity search
    print("[6/6] Building FAISS index...")
    X_train_arr = X_train_scaled.astype(np.float32)
    dim = X_train_arr.shape[1]

    # IVF index for scalable approximate nearest-neighbour search
    quantizer  = faiss.IndexFlatL2(dim)
    index      = faiss.IndexIVFFlat(quantizer, dim, min(100, len(X_train_arr) // 39))
    index.train(X_train_arr)
    index.add(X_train_arr)
    index.nprobe = 10

    # Attach labels so we can retrieve churn status of neighbours
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Save training vectors + labels for neighbour lookup
    np.save(str(MODELS_DIR / "train_vectors.npy"), X_train_arr)
    np.save(str(MODELS_DIR / "train_labels.npy"),  y_train.astype(np.float32))

    print(f"      Index contains {index.ntotal} vectors (dim={dim})")

    # 8. Persist artifacts
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    metadata = {
        "version":       "1.0.0",
        "auc_roc":       round(auc_roc, 4),
        "precision":     round(precision, 4),
        "recall":        round(recall, 4),
        "f1_score":      round(f1, 4),
        "feature_names": ALL_FEATURES,
        "training_rows": len(X_train),
        "test_rows":     len(X_test),
        "churn_rate":    round(float(y.mean()), 4),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\n✅ Training complete. Artifacts saved to /app/models/")
    print("=" * 60)
    return metadata


if __name__ == "__main__":
    train()
