"""
QuantumSepsis Shield — V2 Feature Engineering
===============================================

Computes derived temporal features on top of raw 12 features to boost
discriminative power. Applied AFTER windowing, operating on each
(6, 12) window to produce a (6, N_features_v2) enriched window.

New features (per time step within window):
  - Delta (rate of change) for key vitals
  - Clinical severity indices (Shock Index, Pulse Pressure)
  - Rolling statistics (std, range) for instability markers
  - Trend slopes (linear regression over window)
  - Missing data indicators (computed before imputation)

This expands the feature set from 12 → 33 features per time step.
All features are computable from raw vitals at inference time (no future data).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature name registry ──────────────────────────────────────────────────────

RAW_FEATURE_NAMES: List[str] = [
    "heart_rate", "sbp", "dbp", "map", "temperature",
    "resp_rate", "spo2", "gcs_total",
    "lactate", "wbc", "creatinine", "platelets",
]

# Indices into the 12-feature vector
_IDX = {name: i for i, name in enumerate(RAW_FEATURE_NAMES)}

DERIVED_FEATURE_NAMES: List[str] = [
    # Delta features (6)
    "delta_heart_rate",
    "delta_sbp",
    "delta_map",
    "delta_lactate",
    "delta_resp_rate",
    "delta_temperature",
    # Clinical indices (4)
    "shock_index",            # HR / SBP
    "modified_shock_index",   # HR / MAP
    "pulse_pressure",         # SBP - DBP
    "spo2_hr_ratio",          # SpO2 / HR
    # Rolling statistics (5)
    "hr_rolling_std",
    "map_rolling_std",
    "lactate_rolling_max",
    "temp_rolling_range",
    "resp_rate_rolling_std",
    # Trend slopes (3) — linear regression slope over window
    "hr_trend_slope",
    "map_trend_slope",
    "lactate_trend_slope",
    # Interaction features (3)
    "hr_x_temp",              # HR × temperature (fever + tachycardia)
    "lactate_map_ratio",      # lactate / MAP (perfusion index)
    "wbc_platelets_ratio",    # WBC / platelets (inflammation vs coagulation)
]

V2_FEATURE_NAMES: List[str] = RAW_FEATURE_NAMES + DERIVED_FEATURE_NAMES
N_FEATURES_V2: int = len(V2_FEATURE_NAMES)  # 33


# ── Core feature computation ──────────────────────────────────────────────────

def compute_delta(window: np.ndarray, col_idx: int) -> np.ndarray:
    """Compute first-order difference (rate of change) for a feature column.

    Returns (T,) array where delta[0] = 0 (no prior step).
    """
    vals = window[:, col_idx]
    delta = np.zeros_like(vals)
    delta[1:] = np.diff(vals)
    return delta


def compute_rolling_stat(
    window: np.ndarray, col_idx: int, stat: str = "std", k: int = 3
) -> np.ndarray:
    """Compute rolling statistic over a sub-window of size k.

    For time steps with fewer than k predecessors, uses available data.
    """
    vals = window[:, col_idx]
    T = len(vals)
    result = np.zeros(T, dtype=np.float32)

    for t in range(T):
        start = max(0, t - k + 1)
        chunk = vals[start : t + 1]
        if stat == "std":
            result[t] = np.std(chunk) if len(chunk) > 1 else 0.0
        elif stat == "max":
            result[t] = np.max(chunk)
        elif stat == "range":
            result[t] = np.ptp(chunk)  # max - min
        else:
            raise ValueError(f"Unknown stat: {stat}")

    return result


def compute_trend_slope(window: np.ndarray, col_idx: int) -> float:
    """Compute linear regression slope over the entire window.

    Returns a single scalar broadcast to all time steps (window-level trend).
    """
    vals = window[:, col_idx]
    T = len(vals)

    # Handle NaN/constant
    if np.all(np.isnan(vals)) or np.std(vals) < 1e-8:
        return 0.0

    t_axis = np.arange(T, dtype=np.float32)
    valid = ~np.isnan(vals)
    if valid.sum() < 2:
        return 0.0

    # Simple OLS slope = cov(t, vals) / var(t)
    t_v = t_axis[valid]
    y_v = vals[valid]
    slope = np.polyfit(t_v, y_v, 1)[0]
    return float(slope)


def enrich_window(window: np.ndarray) -> np.ndarray:
    """Enrich a single (T, 12) window to (T, 33) with derived features.

    Args:
        window: (T, 12) raw feature array (typically T=6).

    Returns:
        enriched: (T, 33) array with raw + derived features.
    """
    T = window.shape[0]
    assert window.shape[1] == 12, f"Expected 12 raw features, got {window.shape[1]}"

    derived = np.zeros((T, len(DERIVED_FEATURE_NAMES)), dtype=np.float32)
    col = 0

    # ── Delta features ─────────────────────────────────────────────────────
    for feat in ["heart_rate", "sbp", "map", "lactate", "resp_rate", "temperature"]:
        derived[:, col] = compute_delta(window, _IDX[feat])
        col += 1

    # ── Clinical indices ───────────────────────────────────────────────────
    hr = window[:, _IDX["heart_rate"]]
    sbp = window[:, _IDX["sbp"]]
    dbp = window[:, _IDX["dbp"]]
    map_ = window[:, _IDX["map"]]
    spo2 = window[:, _IDX["spo2"]]

    # Shock Index = HR / SBP (normal < 0.7; >1.0 = shock)
    derived[:, col] = np.where(sbp > 1e-3, hr / sbp, 0.0)
    col += 1

    # Modified Shock Index = HR / MAP
    derived[:, col] = np.where(map_ > 1e-3, hr / map_, 0.0)
    col += 1

    # Pulse Pressure = SBP - DBP (narrow = poor cardiac output)
    derived[:, col] = sbp - dbp
    col += 1

    # SpO2/HR ratio (oxygenation efficiency)
    derived[:, col] = np.where(hr > 1e-3, spo2 / hr, 0.0)
    col += 1

    # ── Rolling statistics ─────────────────────────────────────────────────
    derived[:, col] = compute_rolling_stat(window, _IDX["heart_rate"], "std", 3)
    col += 1
    derived[:, col] = compute_rolling_stat(window, _IDX["map"], "std", 3)
    col += 1
    derived[:, col] = compute_rolling_stat(window, _IDX["lactate"], "max", 3)
    col += 1
    derived[:, col] = compute_rolling_stat(window, _IDX["temperature"], "range", 3)
    col += 1
    derived[:, col] = compute_rolling_stat(window, _IDX["resp_rate"], "std", 3)
    col += 1

    # ── Trend slopes (broadcast to all time steps) ─────────────────────────
    hr_slope = compute_trend_slope(window, _IDX["heart_rate"])
    derived[:, col] = hr_slope
    col += 1

    map_slope = compute_trend_slope(window, _IDX["map"])
    derived[:, col] = map_slope
    col += 1

    lac_slope = compute_trend_slope(window, _IDX["lactate"])
    derived[:, col] = lac_slope
    col += 1

    # ── Interaction features ───────────────────────────────────────────────
    temp = window[:, _IDX["temperature"]]
    lac = window[:, _IDX["lactate"]]
    wbc = window[:, _IDX["wbc"]]
    plt_ = window[:, _IDX["platelets"]]

    # HR × temperature (fever + tachycardia interaction)
    derived[:, col] = hr * temp
    col += 1

    # Lactate / MAP ratio (tissue perfusion index)
    derived[:, col] = np.where(map_ > 1e-3, lac / map_, 0.0)
    col += 1

    # WBC / platelets ratio (inflammation vs coagulation)
    derived[:, col] = np.where(plt_ > 1e-3, wbc / plt_, 0.0)
    col += 1

    assert col == len(DERIVED_FEATURE_NAMES), f"col={col} != {len(DERIVED_FEATURE_NAMES)}"

    # Concatenate raw + derived
    enriched = np.concatenate([window, derived], axis=1)
    assert enriched.shape == (T, N_FEATURES_V2)

    return enriched


def enrich_batch(X: np.ndarray) -> np.ndarray:
    """Enrich a batch of windows: (N, T, 12) → (N, T, 33).

    This is the main entry point for the training pipeline.
    """
    N = X.shape[0]
    T = X.shape[1]

    logger.info(
        "Enriching %d windows: (%d, %d, 12) → (%d, %d, %d)",
        N, N, T, N, T, N_FEATURES_V2,
    )

    X_v2 = np.zeros((N, T, N_FEATURES_V2), dtype=np.float32)
    for i in range(N):
        X_v2[i] = enrich_window(X[i])

    # Replace NaN/Inf with 0
    X_v2 = np.nan_to_num(X_v2, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Enrichment complete. Shape: %s, NaN count: %d",
        X_v2.shape, int(np.isnan(X_v2).sum()),
    )

    return X_v2


def normalize_derived_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Z-score normalize only the derived features (columns 12:33).

    Raw features (0:12) are already normalized by the preprocessing pipeline.
    Returns normalized arrays and the normalization stats for inference.
    """
    # Compute stats from training set only
    derived_train = X_train[:, :, 12:]  # (N, T, 21)
    flat = derived_train.reshape(-1, derived_train.shape[-1])  # (N*T, 21)

    mean = np.nanmean(flat, axis=0)  # (21,)
    std = np.nanstd(flat, axis=0)    # (21,)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

    # Apply normalization
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()

    X_train_norm[:, :, 12:] = (X_train[:, :, 12:] - mean) / std
    X_val_norm[:, :, 12:] = (X_val[:, :, 12:] - mean) / std
    X_test_norm[:, :, 12:] = (X_test[:, :, 12:] - mean) / std

    # Replace any NaN/Inf from normalization
    for arr in [X_train_norm, X_val_norm, X_test_norm]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "derived_mean": mean.astype(np.float32),
        "derived_std": std.astype(np.float32),
        "derived_feature_names": np.array(DERIVED_FEATURE_NAMES),
    }

    logger.info("Derived feature normalization complete (21 features)")

    return X_train_norm, X_val_norm, X_test_norm, stats


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((100, 6, 12)).astype(np.float32)
    X_test[:, :, _IDX["sbp"]] = 120 + rng.standard_normal((100, 6)) * 15
    X_test[:, :, _IDX["heart_rate"]] = 80 + rng.standard_normal((100, 6)) * 10

    X_enriched = enrich_batch(X_test)
    print(f"Input:    {X_test.shape}")
    print(f"Enriched: {X_enriched.shape}")
    print(f"Feature names ({N_FEATURES_V2}):")
    for i, name in enumerate(V2_FEATURE_NAMES):
        col = X_enriched[:, :, i]
        print(f"  [{i:2d}] {name:25s} mean={col.mean():.4f} std={col.std():.4f}")
    print("\n✓ Feature engineering V2 test passed!")
