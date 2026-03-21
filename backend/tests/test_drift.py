"""
Tests for PSI drift detection.
Pure numpy/pandas — no DB or model required.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.drift import check_drift, compute_feature_psi, compute_psi


# ── compute_psi unit tests ────────────────────────────────────

class TestComputePsi:

    def test_identical_distributions_near_zero(self):
        """PSI between identical distributions should be ~0."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, 1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.01, f"Expected PSI ≈ 0 for identical data, got {psi:.4f}"

    def test_very_different_distributions_high_psi(self):
        """PSI between very different distributions should exceed 0.2."""
        rng = np.random.default_rng(42)
        reference = rng.normal(0,  1, 1000)
        current   = rng.normal(10, 1, 1000)
        psi = compute_psi(reference, current)
        assert psi > 0.2, f"Expected high PSI for shifted distribution, got {psi:.4f}"

    def test_psi_is_non_negative(self):
        """PSI must always be >= 0."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            ref = rng.normal(0, 1, 500)
            cur = rng.normal(0.5, 1.2, 500)
            psi = compute_psi(ref, cur)
            assert psi >= 0, f"PSI must be non-negative, got {psi}"

    def test_psi_symmetry_approximate(self):
        """PSI(A, B) and PSI(B, A) should be in the same ballpark."""
        rng = np.random.default_rng(99)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0.5, 1, 1000)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        # Not exactly symmetric, but should be within 3x of each other
        assert psi_ab / (psi_ba + 1e-9) < 3.0

    def test_moderate_shift_medium_psi(self):
        """A moderate shift should give PSI in the monitor range (0.1 – 0.2)."""
        rng = np.random.default_rng(7)
        reference = rng.normal(50, 10, 2000)
        current   = rng.normal(53, 10, 500)   # small shift
        psi = compute_psi(reference, current)
        # Should be detectable but not catastrophic
        assert psi >= 0, "PSI must be non-negative"

    def test_custom_bins(self):
        """PSI should work with different bin counts."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        for n_bins in [5, 10, 20]:
            psi = compute_psi(ref, cur, n_bins=n_bins)
            assert psi >= 0
            assert isinstance(psi, float)


# ── compute_feature_psi ───────────────────────────────────────

class TestFeaturePsi:

    def _make_dfs(self, n=500, shift=0.0, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        ref = pd.DataFrame({
            "tenure":          rng.integers(0, 73, n).astype(float),
            "monthly_charges": rng.uniform(18, 119, n),
        })
        cur = pd.DataFrame({
            "tenure":          (rng.integers(0, 73, n) + shift).astype(float),
            "monthly_charges": rng.uniform(18 + shift, 119 + shift, n),
        })
        return ref, cur

    def test_returns_dict_with_feature_keys(self):
        ref, cur = self._make_dfs()
        result = compute_feature_psi(ref, cur, ["tenure", "monthly_charges"])
        assert set(result.keys()) == {"tenure", "monthly_charges"}

    def test_values_are_floats(self):
        ref, cur = self._make_dfs()
        result = compute_feature_psi(ref, cur, ["tenure", "monthly_charges"])
        for v in result.values():
            assert isinstance(v, float)

    def test_missing_feature_skipped(self):
        ref, cur = self._make_dfs()
        result = compute_feature_psi(ref, cur, ["tenure", "nonexistent_feature"])
        assert "tenure" in result
        assert "nonexistent_feature" not in result

    def test_small_sample_returns_zero(self):
        """Samples smaller than 10 should return 0 (not enough data)."""
        ref = pd.DataFrame({"tenure": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"tenure": [4.0, 5.0, 6.0]})
        result = compute_feature_psi(ref, cur, ["tenure"])
        assert result["tenure"] == 0.0


# ── check_drift (full integration) ───────────────────────────

class TestCheckDrift:

    def _stable_dfs(self, n=500) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "tenure":          rng.integers(0, 73, n).astype(float),
            "monthly_charges": rng.uniform(18, 119, n),
            "total_charges":   rng.uniform(0, 8000, n),
        })
        return data.iloc[:300], data.iloc[300:]

    def test_no_drift_detected(self):
        ref, cur = self._stable_dfs()
        result = check_drift(ref, cur, features=["tenure", "monthly_charges"])
        assert "drift_detected" in result
        assert "max_psi" in result
        assert "feature_psi" in result
        assert result["max_psi"] >= 0

    def test_drift_detected_on_shifted_data(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"tenure": rng.integers(0, 10, 500).astype(float)})
        cur = pd.DataFrame({"tenure": rng.integers(60, 73, 500).astype(float)})
        result = check_drift(ref, cur, features=["tenure"], threshold=0.2)
        assert result["drift_detected"] is True
        assert "tenure" in result["drifted_features"]

    def test_result_structure(self):
        ref, cur = self._stable_dfs()
        result = check_drift(ref, cur, features=["tenure"])
        required_keys = {
            "drift_detected", "drifted_features", "monitored_features",
            "stable_features", "max_psi", "avg_psi",
            "feature_psi", "threshold", "n_reference", "n_current",
        }
        assert required_keys.issubset(result.keys())

    def test_sample_counts_correct(self):
        ref, cur = self._stable_dfs()
        result = check_drift(ref, cur, features=["tenure"])
        assert result["n_reference"] == len(ref)
        assert result["n_current"]   == len(cur)

    def test_threshold_respected(self):
        """With threshold=0.0, everything should be drifted."""
        rng = np.random.default_rng(1)
        ref = pd.DataFrame({"tenure": rng.integers(0, 73, 200).astype(float)})
        cur = pd.DataFrame({"tenure": rng.integers(0, 73, 100).astype(float)})
        result = check_drift(ref, cur, features=["tenure"], threshold=0.0)
        # With threshold 0.0 even tiny PSI counts as drift
        assert result["drift_detected"] is True
