"""
Data drift detection using Population Stability Index (PSI).

PSI < 0.1  → no significant drift
PSI < 0.2  → moderate drift (monitor)
PSI >= 0.2 → significant drift → trigger retraining alert

This is the metric interviewers ask about when you claim
drift detection. Know it cold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index between two distributions.

    Args:
        reference: Training distribution (baseline)
        current:   Recent inference distribution
        n_bins:    Number of histogram bins
        epsilon:   Smoothing to avoid log(0)

    Returns:
        PSI score (float)
    """
    # Compute bin edges from reference distribution
    breakpoints = np.linspace(
        np.percentile(reference, 0),
        np.percentile(reference, 100),
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    # Convert to proportions
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (len(current)   + epsilon * n_bins)

    # PSI = Σ (cur% - ref%) * ln(cur% / ref%)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_feature_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Compute PSI for each feature between reference and current data."""
    results = {}
    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref = reference_df[feat].dropna().values
        cur = current_df[feat].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            results[feat] = 0.0
            continue
        results[feat] = compute_psi(ref, cur)
    return results


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.2,
) -> dict:
    """
    Run full drift check across all features.

    Returns a summary dict with per-feature PSI and overall drift flag.
    """
    psi_scores = compute_feature_psi(reference_df, current_df, features)

    drifted = {k: v for k, v in psi_scores.items() if v >= threshold}
    monitored = {k: v for k, v in psi_scores.items() if 0.1 <= v < threshold}

    return {
        "drift_detected": len(drifted) > 0,
        "drifted_features": drifted,
        "monitored_features": monitored,
        "stable_features": {k: v for k, v in psi_scores.items() if v < 0.1},
        "max_psi": max(psi_scores.values()) if psi_scores else 0.0,
        "avg_psi": float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0,
        "feature_psi": psi_scores,
        "threshold": threshold,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
    }
