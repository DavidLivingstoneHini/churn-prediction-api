"""
Inference engine — loads model artifacts once at startup,
exposes predict() and similar_customers() for fast inference.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.ml.features import ALL_FEATURES, engineer_single

MODELS_DIR = Path("/app/models")


@dataclass
class PredictionResult:
    churn_probability: float
    risk_level: str          # low | medium | high
    model_version: str
    inference_time_ms: int
    feature_values: dict
    similar_customers: list[dict]


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "medium"
    return "high"


class ChurnPredictor:
    """Singleton — loaded once on startup, used across all requests."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._index = None
        self._train_vectors: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._metadata: dict = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        model_path  = MODELS_DIR / "churn_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        index_path  = MODELS_DIR / "faiss.index"
        meta_path   = MODELS_DIR / "model_metadata.json"
        vec_path    = MODELS_DIR / "train_vectors.npy"
        lbl_path    = MODELS_DIR / "train_labels.npy"

        if not model_path.exists():
            raise RuntimeError(
                "Model artifacts not found. Run `python -m app.ml.train` first."
            )

        with open(model_path,  "rb") as f: self._model  = pickle.load(f)
        with open(scaler_path, "rb") as f: self._scaler = pickle.load(f)

        self._index         = faiss.read_index(str(index_path))
        self._train_vectors = np.load(str(vec_path))
        self._train_labels  = np.load(str(lbl_path))
        self._metadata      = json.loads(meta_path.read_text())
        self._loaded        = True

    def predict(self, features: dict, top_k: int = 5) -> PredictionResult:
        if not self._loaded:
            self.load()

        t0 = time.monotonic()

        # Engineer + scale
        raw = engineer_single(features)
        scaled = self._scaler.transform(raw).astype(np.float32)

        # Predict probability
        prob = float(self._model.predict_proba(scaled)[0, 1])

        # FAISS similarity search
        distances, indices = self._index.search(scaled, top_k + 1)
        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._train_labels):
                continue
            similar.append({
                "similarity_score": round(float(1 / (1 + dist)), 4),
                "churn_label": int(self._train_labels[idx]),
                "distance": round(float(dist), 4),
            })
        similar = similar[:top_k]

        elapsed = int((time.monotonic() - t0) * 1000)

        return PredictionResult(
            churn_probability=round(prob, 4),
            risk_level=_risk_level(prob),
            model_version=self._metadata.get("version", "1.0.0"),
            inference_time_ms=elapsed,
            feature_values={k: v for k, v in zip(ALL_FEATURES, raw[0].tolist())},
            similar_customers=similar,
        )

    def predict_batch(self, records: list[dict]) -> list[PredictionResult]:
        return [self.predict(r) for r in records]

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def version(self) -> str:
        return self._metadata.get("version", "unknown")


# Module-level singleton
predictor = ChurnPredictor()
