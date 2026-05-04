"""
QuantumSepsis Shield — V1 Enhanced Feature Engineering
=======================================================

Adds 27 temporal features to V1's 12 raw features → 39 total features.

CRITICAL LESSONS FROM V2 FAILURE:
  1. Compute derived features from RAW (denormalized) values
  2. Normalize ALL 39 features together with consistent statistics
  3. Never mix normalized and raw scales in computations

New Features (27):
  - Deltas (12): hour-to-hour changes for all 12 features
  - Trends (6): linear slopes for HR, MAP, Temp, RR, Lactate, GCS
  - Rolling stats (3): rolling mean for HR, MAP, Lactate
  - Clinical indices (3): Shock Index, MEWS RR score, Lactate clearance
  - Interactions (3): HR×Temp, MAP×Lactate, WBC×Temp

Total: 12 + 12 + 6 + 3 + 3 + 3 = 39 ✓

Expected gain: +1-2% AUROC over V1 baseline
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config

logger = logging.getLogger(__name__)


class V1EnhancedFeatureEngineer:
    """Enhanced feature engineering for V1 model.

    Takes normalized 12-feature windows, denormalizes, enriches to 39 features,
    then normalizes all together.
    """

    def __init__(
        self,
        original_norm_stats: Dict[str, Dict[str, float]],
        config: Config = None
    ):
        """Initialize feature engineer.

        Args:
            original_norm_stats: Dict with 'train_mean', 'train_std', 'train_median'
                                 for the original 12 features
            config: Configuration object
        """
        if config is None:
            config = get_default_config()

        self.config = config
        self.feature_names = config.data.feature_names  # Original 12 features

        # Original normalization stats (for denormalization)
        self.orig_mean = np.array([original_norm_stats['train_mean'][f]
                                   for f in self.feature_names], dtype=np.float32)
        self.orig_std = np.array([original_norm_stats['train_std'][f]
                                  for f in self.feature_names], dtype=np.float32)

        # New normalization stats (computed from enriched features)
        self.enriched_mean = None
        self.enriched_std = None

        # Feature indices for easy access
        self.hr_idx = self.feature_names.index("heart_rate")
        self.sbp_idx = self.feature_names.index("sbp")
        self.map_idx = self.feature_names.index("map")
        self.temp_idx = self.feature_names.index("temperature")
        self.lactate_idx = self.feature_names.index("lactate")
        self.wbc_idx = self.feature_names.index("wbc")
        self.rr_idx = self.feature_names.index("resp_rate")
        self.gcs_idx = self.feature_names.index("gcs_total")

    def fit(self, X_train_normalized: np.ndarray) -> None:
        """Compute normalization statistics for enriched features.

        Args:
            X_train_normalized: (N, 6, 12) normalized training data
        """
        logger.info("Computing enriched feature statistics from training data...")

        # Denormalize
        X_raw = self._denormalize(X_train_normalized)

        # Enrich
        X_enriched_raw = self._enrich_batch(X_raw)

        # Compute statistics
        # Reshape to (N*6, 39) for per-feature statistics
        X_flat = X_enriched_raw.reshape(-1, X_enriched_raw.shape[-1])

        self.enriched_mean = np.nanmean(X_flat, axis=0).astype(np.float32)
        self.enriched_std = np.nanstd(X_flat, axis=0).astype(np.float32)

        # Prevent division by zero
        self.enriched_std = np.where(self.enriched_std < 1e-8, 1.0, self.enriched_std)

        logger.info(f"Enriched feature statistics computed:")
        logger.info(f"  Mean range: [{self.enriched_mean.min():.2f}, {self.enriched_mean.max():.2f}]")
        logger.info(f"  Std range: [{self.enriched_std.min():.2f}, {self.enriched_std.max():.2f}]")

    def transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """Transform normalized 12-feature windows to normalized 39-feature windows.

        Args:
            X_normalized: (N, 6, 12) normalized input

        Returns:
            (N, 6, 39) normalized enriched features
        """
        assert self.enriched_mean is not None, "Must call fit() first"

        # Step 1: Denormalize to raw scale
        X_raw = self._denormalize(X_normalized)

        # Step 2: Enrich features
        X_enriched_raw = self._enrich_batch(X_raw)

        # Step 3: Normalize all 39 features together
        X_enriched_normalized = self._normalize_enriched(X_enriched_raw)

        return X_enriched_normalized

    def _denormalize(self, X_normalized: np.ndarray) -> np.ndarray:
        """Denormalize from z-scores back to raw scale.

        Args:
            X_normalized: (N, 6, 12)

        Returns:
            X_raw: (N, 6, 12) in original units
        """
        X_raw = X_normalized * self.orig_std[None, None, :] + self.orig_mean[None, None, :]
        return X_raw

    def _normalize_enriched(self, X_enriched_raw: np.ndarray) -> np.ndarray:
        """Normalize enriched features using computed statistics.

        Args:
            X_enriched_raw: (N, 6, 39)

        Returns:
            X_enriched_normalized: (N, 6, 39)
        """
        X_norm = (X_enriched_raw - self.enriched_mean[None, None, :]) / self.enriched_std[None, None, :]
        return X_norm

    def _enrich_batch(self, X_raw: np.ndarray) -> np.ndarray:
        """Enrich a batch of raw windows with temporal features.

        Args:
            X_raw: (N, 6, 12) raw features

        Returns:
            X_enriched: (N, 6, 39) enriched features
        """
        N, T, F = X_raw.shape

        # Initialize output: 12 original + 27 new = 39
        X_enriched = np.zeros((N, T, 39), dtype=np.float32)

        # Copy original 12 features
        X_enriched[:, :, :12] = X_raw

        # Feature index counter
        feat_idx = 12

        # 1. DELTAS (12 features): hour-to-hour changes
        # For each timestep, compute delta from previous timestep
        deltas = np.zeros_like(X_raw)
        deltas[:, 1:, :] = X_raw[:, 1:, :] - X_raw[:, :-1, :]
        deltas[:, 0, :] = 0.0  # First timestep has no delta
        X_enriched[:, :, feat_idx:feat_idx+12] = deltas
        feat_idx += 12

        # 2. TRENDS (6 features): linear slopes for critical vitals
        # Compute slope over the 6-hour window for: HR, MAP, Temp, RR, Lactate, GCS
        critical_indices = [self.hr_idx, self.map_idx, self.temp_idx,
                           self.rr_idx, self.lactate_idx, self.gcs_idx]

        trends = self._compute_trends(X_raw, critical_indices)
        # Broadcast trend (scalar per window) to all timesteps
        X_enriched[:, :, feat_idx:feat_idx+6] = trends[:, None, :]
        feat_idx += 6

        # 3. ROLLING STATS (3 features): rolling mean for HR, MAP, Lactate
        # Compute cumulative statistics up to each timestep
        for i, var_idx in enumerate([self.hr_idx, self.map_idx, self.lactate_idx]):
            # Rolling mean
            rolling_mean = self._compute_rolling_mean(X_raw[:, :, var_idx])
            X_enriched[:, :, feat_idx] = rolling_mean
            feat_idx += 1

        # 4. CLINICAL INDICES (3 features)
        # Shock Index: HR / SBP (normal < 0.7, shock > 1.0)
        shock_index = np.where(
            X_raw[:, :, self.sbp_idx] > 1e-3,
            X_raw[:, :, self.hr_idx] / X_raw[:, :, self.sbp_idx],
            0.0
        )
        X_enriched[:, :, feat_idx] = shock_index
        feat_idx += 1

        # MEWS component: RR score (0-3 based on respiratory rate)
        rr_score = self._compute_rr_score(X_raw[:, :, self.rr_idx])
        X_enriched[:, :, feat_idx] = rr_score
        feat_idx += 1

        # Lactate clearance: (first_lactate - current_lactate) / first_lactate
        first_lactate = X_raw[:, 0:1, self.lactate_idx]  # (N, 1)
        lactate_clearance = np.where(
            first_lactate > 1e-3,
            (first_lactate - X_raw[:, :, self.lactate_idx]) / first_lactate,
            0.0
        )
        X_enriched[:, :, feat_idx] = lactate_clearance
        feat_idx += 1

        # 5. INTERACTION TERMS (3 features)
        # HR × Temperature (fever + tachycardia)
        X_enriched[:, :, feat_idx] = X_raw[:, :, self.hr_idx] * X_raw[:, :, self.temp_idx]
        feat_idx += 1

        # MAP × Lactate (perfusion + tissue hypoxia)
        X_enriched[:, :, feat_idx] = X_raw[:, :, self.map_idx] * X_raw[:, :, self.lactate_idx]
        feat_idx += 1

        # WBC × Temperature (infection markers)
        X_enriched[:, :, feat_idx] = X_raw[:, :, self.wbc_idx] * X_raw[:, :, self.temp_idx]
        feat_idx += 1

        assert feat_idx == 39, f"Expected 39 features, got {feat_idx}"

        # Replace NaN with 0
        X_enriched = np.nan_to_num(X_enriched, nan=0.0, posinf=0.0, neginf=0.0)

        return X_enriched

    def _compute_trends(self, X: np.ndarray, feature_indices: list) -> np.ndarray:
        """Compute linear trend (slope) over the 6-hour window.

        Args:
            X: (N, 6, F) raw features
            feature_indices: List of feature indices to compute trends for

        Returns:
            trends: (N, len(feature_indices)) slopes
        """
        N, T, F = X.shape
        n_features = len(feature_indices)
        trends = np.zeros((N, n_features), dtype=np.float32)

        t = np.arange(T, dtype=np.float32)

        for i, feat_idx in enumerate(feature_indices):
            for n in range(N):
                vals = X[n, :, feat_idx]
                valid = ~np.isnan(vals)
                if valid.sum() >= 2:
                    # Fit linear regression: y = slope * t + intercept
                    slope = np.polyfit(t[valid], vals[valid], 1)[0]
                    trends[n, i] = slope

        return trends

    def _compute_rolling_mean(self, x: np.ndarray) -> np.ndarray:
        """Compute cumulative mean up to each timestep.

        Args:
            x: (N, T) single feature values

        Returns:
            rolling_mean: (N, T)
        """
        N, T = x.shape
        rolling_mean = np.zeros_like(x)

        for t in range(T):
            rolling_mean[:, t] = np.nanmean(x[:, :t+1], axis=1)

        return rolling_mean

    def _compute_rolling_std(self, x: np.ndarray) -> np.ndarray:
        """Compute cumulative std up to each timestep.

        Args:
            x: (N, T) single feature values

        Returns:
            rolling_std: (N, T)
        """
        N, T = x.shape
        rolling_std = np.zeros_like(x)

        for t in range(T):
            rolling_std[:, t] = np.nanstd(x[:, :t+1], axis=1)

        return rolling_std

    def _compute_rr_score(self, rr: np.ndarray) -> np.ndarray:
        """Compute MEWS respiratory rate score.

        Score:
            0: RR 9-14
            1: RR 15-20
            2: RR 21-29 or <9
            3: RR ≥30

        Args:
            rr: (N, T) respiratory rate values

        Returns:
            score: (N, T)
        """
        score = np.zeros_like(rr)
        score = np.where((rr >= 9) & (rr <= 14), 0, score)
        score = np.where((rr >= 15) & (rr <= 20), 1, score)
        score = np.where(((rr >= 21) & (rr <= 29)) | (rr < 9), 2, score)
        score = np.where(rr >= 30, 3, score)
        return score

    def get_feature_names(self) -> list:
        """Get names of all 39 enriched features."""
        names = self.feature_names.copy()  # Original 12

        # Deltas
        names += [f"{f}_delta" for f in self.feature_names]

        # Trends
        names += ["hr_trend", "map_trend", "temp_trend", "rr_trend", "lactate_trend", "gcs_trend"]

        # Rolling stats (only mean, not std)
        names += ["hr_rolling_mean", "map_rolling_mean", "lactate_rolling_mean"]

        # Clinical indices
        names += ["shock_index", "mews_rr_score", "lactate_clearance"]

        # Interactions
        names += ["hr_temp_interaction", "map_lactate_interaction", "wbc_temp_interaction"]

        return names


def test_feature_engineering():
    """Test the enhanced feature engineering with synthetic data."""
    print("Testing V1 Enhanced Feature Engineering")
    print("=" * 60)

    # Create synthetic normalized data
    np.random.seed(42)
    N = 100
    X_normalized = np.random.randn(N, 6, 12).astype(np.float32)

    # Create mock normalization stats
    config = get_default_config()
    feature_names = config.data.feature_names

    original_stats = {
        'train_mean': {f: np.random.uniform(50, 100) for f in feature_names},
        'train_std': {f: np.random.uniform(5, 20) for f in feature_names},
        'train_median': {f: np.random.uniform(50, 100) for f in feature_names},
    }

    # Initialize engineer
    engineer = V1EnhancedFeatureEngineer(original_stats, config)

    # Fit on training data
    engineer.fit(X_normalized)

    # Transform
    X_enriched = engineer.transform(X_normalized)

    print(f"Input shape:  {X_normalized.shape}")
    print(f"Output shape: {X_enriched.shape}")
    print(f"Expected:     (100, 6, 39)")

    assert X_enriched.shape == (N, 6, 39), f"Wrong shape: {X_enriched.shape}"

    # Check for NaN
    n_nan = np.isnan(X_enriched).sum()
    print(f"\nNaN count: {n_nan} (should be 0)")
    assert n_nan == 0, "Output contains NaN values!"

    # Check value ranges
    print(f"\nValue ranges:")
    print(f"  Min: {X_enriched.min():.3f}")
    print(f"  Max: {X_enriched.max():.3f}")
    print(f"  Mean: {X_enriched.mean():.3f}")
    print(f"  Std: {X_enriched.std():.3f}")

    # Print feature names
    print(f"\nEnriched feature names ({len(engineer.get_feature_names())}):")
    for i, name in enumerate(engineer.get_feature_names()):
        print(f"  {i:2d}. {name}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_feature_engineering()
