"""
Tests for the ML inference engine.
Uses fake artifacts written by conftest.py — no real model needed.
"""
import numpy as np
import pytest

from app.ml.predictor import ChurnPredictor, _risk_level


# ── _risk_level helper ────────────────────────────────────────

class TestRiskLevel:

    def test_low_risk(self):
        assert _risk_level(0.0)  == "low"
        assert _risk_level(0.1)  == "low"
        assert _risk_level(0.29) == "low"

    def test_medium_risk(self):
        assert _risk_level(0.30) == "medium"
        assert _risk_level(0.45) == "medium"
        assert _risk_level(0.59) == "medium"

    def test_high_risk(self):
        assert _risk_level(0.60) == "high"
        assert _risk_level(0.75) == "high"
        assert _risk_level(1.0)  == "high"

    def test_boundary_values(self):
        """Boundary conditions must be correctly classified."""
        assert _risk_level(0.3) == "medium"
        assert _risk_level(0.6) == "high"


# ── ChurnPredictor ────────────────────────────────────────────

def _make_features() -> dict:
    return {
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


class TestChurnPredictor:

    @pytest.fixture(autouse=True)
    def predictor(self):
        p = ChurnPredictor()
        p.load()
        return p

    def test_predictor_loads(self, predictor):
        assert predictor._loaded is True

    def test_prediction_returns_result(self, predictor):
        result = predictor.predict(_make_features())
        assert result is not None

    def test_churn_probability_range(self, predictor):
        result = predictor.predict(_make_features())
        assert 0.0 <= result.churn_probability <= 1.0

    def test_risk_level_valid(self, predictor):
        result = predictor.predict(_make_features())
        assert result.risk_level in ("low", "medium", "high")

    def test_model_version_set(self, predictor):
        result = predictor.predict(_make_features())
        assert result.model_version == "1.0.0"

    def test_inference_time_positive(self, predictor):
        result = predictor.predict(_make_features())
        assert result.inference_time_ms >= 0

    def test_similar_customers_returned(self, predictor):
        result = predictor.predict(_make_features(), top_k=3)
        assert isinstance(result.similar_customers, list)
        assert len(result.similar_customers) <= 3

    def test_similar_customer_fields(self, predictor):
        result = predictor.predict(_make_features(), top_k=2)
        for sc in result.similar_customers:
            assert "similarity_score" in sc
            assert "churn_label"      in sc
            assert "distance"         in sc
            assert sc["churn_label"] in (0, 1)
            assert 0.0 <= sc["similarity_score"] <= 1.0

    def test_feature_values_returned(self, predictor):
        result = predictor.predict(_make_features())
        assert isinstance(result.feature_values, dict)
        assert len(result.feature_values) == 29

    def test_batch_predict(self, predictor):
        features_list = [_make_features(), _make_features()]
        results = predictor.predict_batch(features_list)
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.churn_probability <= 1.0

    def test_metadata_property(self, predictor):
        meta = predictor.metadata
        assert isinstance(meta, dict)
        assert "version" in meta

    def test_version_property(self, predictor):
        assert predictor.version == "1.0.0"

    def test_high_risk_customer(self, predictor):
        """Customer with high-risk profile should have churn_probability > 0.5."""
        result = predictor.predict(_make_features())
        # The real trained model (not fake) gives ~0.69 for this profile
        assert result.churn_probability > 0.5
        assert result.risk_level in ("medium", "high")

    def test_load_idempotent(self, predictor):
        """Calling load() twice should not raise or reset state."""
        predictor.load()
        assert predictor._loaded is True
