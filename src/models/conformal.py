"""
QuantumSepsis Shield — Conformal Prediction Wrapper
=====================================================

Provides statistically guaranteed confidence intervals on risk scores.

Methods implemented:
  1. Split conformal prediction (standard)
  2. Quantum-Calibrated Conformal Prediction (QCCP) — Novelty 1
     Uses quantum kernel distance as nonconformity score

Configuration:
  - Coverage guarantee: 90% (α = 0.10)
  - Calibration set: 20% of training positives
  - Escalation rule: width > 0.4 → escalate alert tier
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import ConformalConfig, get_default_config

logger = logging.getLogger(__name__)


class ConformalSepsisPredictor:
    """Split conformal prediction wrapper for sepsis risk scores.
    
    Produces prediction intervals with guaranteed coverage:
        P(y_true ∈ [lower, upper]) ≥ 1 - α
    
    The key output is a confidence interval around the risk score.
    Wide intervals indicate uncertainty → trigger alert escalation.
    
    Args:
        config: ConformalConfig with method parameters
    """
    
    def __init__(self, config: Optional[ConformalConfig] = None):
        if config is None:
            config = get_default_config().conformal
        
        self.config = config
        self.alpha = config.alpha
        self.calibrated = False
        
        # Calibration quantile (computed during calibration)
        self.q_alpha: float = 0.0
        
        # Calibration residuals
        self.calibration_scores: Optional[np.ndarray] = None
    
    def calibrate(
        self,
        cal_risk_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Calibrate the conformal predictor using a held-out calibration set.
        
        Computes nonconformity scores on the calibration set and
        determines the quantile threshold for prediction intervals.
        
        Args:
            cal_risk_scores: (N_cal,) predicted risk scores ∈ [0, 1]
            cal_labels: (N_cal,) binary labels
        
        Returns:
            Dictionary with calibration statistics
        """
        n = len(cal_risk_scores)
        assert n > 0, "Calibration set is empty"
        assert len(cal_labels) == n
        
        logger.info(f"Calibrating conformal predictor on {n} samples...")
        
        # Compute nonconformity scores
        # For binary classification: s_i = |y_i - f(x_i)|
        self.calibration_scores = np.abs(cal_labels - cal_risk_scores)
        
        # Compute the (1-α)(1+1/n)-quantile for finite-sample coverage
        quantile_level = min(
            (1 - self.alpha) * (1 + 1/n),
            1.0,
        )
        
        self.q_alpha = float(np.quantile(self.calibration_scores, quantile_level))
        self.calibrated = True
        
        # Compute calibration statistics
        stats = {
            "n_calibration": n,
            "n_positive": int(cal_labels.sum()),
            "n_negative": int((1 - cal_labels).sum()),
            "quantile_level": quantile_level,
            "q_alpha": self.q_alpha,
            "mean_nonconformity": float(self.calibration_scores.mean()),
            "std_nonconformity": float(self.calibration_scores.std()),
        }
        
        logger.info(f"  Calibration complete: q_α = {self.q_alpha:.4f}")
        logger.info(f"  Coverage guarantee: {1 - self.alpha:.0%}")
        
        return stats
    
    def predict(
        self,
        risk_score: float,
    ) -> Tuple[float, float, float, float]:
        """Produce conformal prediction interval for a single sample.
        
        Args:
            risk_score: Point risk estimate ∈ [0, 1]
        
        Returns:
            Tuple of (risk_score, lower_bound, upper_bound, coverage_guarantee)
        """
        assert self.calibrated, "Must call calibrate() before predict()"
        
        lower = max(0.0, risk_score - self.q_alpha)
        upper = min(1.0, risk_score + self.q_alpha)
        coverage = 1.0 - self.alpha
        
        return risk_score, lower, upper, coverage
    
    def predict_batch(
        self,
        risk_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Produce conformal prediction intervals for a batch.
        
        Args:
            risk_scores: (N,) array of risk scores
        
        Returns:
            Tuple of (lower_bounds, upper_bounds, widths)
        """
        assert self.calibrated, "Must call calibrate() before predict_batch()"
        
        lower = np.maximum(0.0, risk_scores - self.q_alpha)
        upper = np.minimum(1.0, risk_scores + self.q_alpha)
        widths = upper - lower
        
        return lower, upper, widths
    
    def should_escalate(self, width: float) -> bool:
        """Check if the prediction interval is too wide → escalate.
        
        Args:
            width: Interval width (upper - lower)
        
        Returns:
            True if width exceeds escalation threshold
        """
        return width > self.config.escalation_width_threshold
    
    def verify_coverage(
        self,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Verify empirical coverage on test set.
        
        Args:
            test_scores: (N_test,) predicted risk scores
            test_labels: (N_test,) ground truth labels
        
        Returns:
            Coverage statistics
        """
        assert self.calibrated
        
        lower, upper, widths = self.predict_batch(test_scores)
        
        # Check coverage: is the true label covered by the interval?
        # For binary: risk_score ∈ [lower, upper] should "cover" the outcome
        # We check if the prediction set contains the correct class
        covered = np.zeros(len(test_labels), dtype=bool)
        for i in range(len(test_labels)):
            if test_labels[i] == 1:
                covered[i] = upper[i] >= 0.5  # Include the positive class
            else:
                covered[i] = lower[i] <= 0.5  # Include the negative class
        
        empirical_coverage = covered.mean()
        
        stats = {
            "empirical_coverage": float(empirical_coverage),
            "target_coverage": 1 - self.alpha,
            "coverage_gap": float(empirical_coverage - (1 - self.alpha)),
            "mean_width": float(widths.mean()),
            "median_width": float(np.median(widths)),
            "max_width": float(widths.max()),
            "min_width": float(widths.min()),
            "pct_escalated": float((widths > self.config.escalation_width_threshold).mean()),
        }
        
        logger.info(f"Coverage verification:")
        logger.info(f"  Empirical coverage: {stats['empirical_coverage']:.1%}")
        logger.info(f"  Target coverage:    {stats['target_coverage']:.1%}")
        logger.info(f"  Mean width:         {stats['mean_width']:.4f}")
        logger.info(f"  % escalated:        {stats['pct_escalated']:.1%}")
        
        return stats


class QuantumCalibratedConformal(ConformalSepsisPredictor):
    """Quantum-Calibrated Conformal Prediction (QCCP) — Novelty 1.
    
    Instead of using |y - f(x)| as the nonconformity score,
    uses quantum kernel distance to learned sepsis centroids:
    
        s(x) = 1 - max_j K(x, c_j)
    
    where c_j are sepsis centroid states in Hilbert space
    and K is the quantum kernel.
    
    This provides tighter prediction sets because the quantum kernel
    captures non-linear structure that inflates classical conformal widths.
    """
    
    def __init__(self, config: Optional[ConformalConfig] = None):
        super().__init__(config)
        self.centroids: Optional[np.ndarray] = None
        self.kernel_fn = None
    
    def set_quantum_kernel(self, kernel_fn, centroids: np.ndarray):
        """Set the quantum kernel function and centroids.
        
        Args:
            kernel_fn: Callable that computes K(x, y) → float
            centroids: (n_centroids, d) array of sepsis centroids
        """
        self.kernel_fn = kernel_fn
        self.centroids = centroids
        logger.info(f"QCCP: Set {len(centroids)} quantum centroids")
    
    def calibrate_quantum(
        self,
        cal_embeddings: np.ndarray,
        cal_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Calibrate using quantum kernel nonconformity scores.
        
        Args:
            cal_embeddings: (N_cal, d) calibration embeddings
            cal_labels: (N_cal,) labels
        
        Returns:
            Calibration statistics
        """
        assert self.kernel_fn is not None, "Must call set_quantum_kernel() first"
        assert self.centroids is not None
        
        n = len(cal_embeddings)
        logger.info(f"QCCP calibrating on {n} samples...")
        
        # Compute quantum nonconformity scores
        nonconformity_scores = np.zeros(n)
        
        for i in range(n):
            # Kernel similarity to each centroid
            max_similarity = 0.0
            for c in self.centroids:
                k_val = self.kernel_fn(
                    cal_embeddings[i:i+1],
                    c.reshape(1, -1)
                )
                if isinstance(k_val, np.ndarray):
                    k_val = k_val[0, 0]
                max_similarity = max(max_similarity, k_val)
            
            nonconformity_scores[i] = 1.0 - max_similarity
        
        self.calibration_scores = nonconformity_scores
        
        # Compute quantile
        quantile_level = min((1 - self.alpha) * (1 + 1/n), 1.0)
        self.q_alpha = float(np.quantile(nonconformity_scores, quantile_level))
        self.calibrated = True
        
        stats = {
            "n_calibration": n,
            "q_alpha_quantum": self.q_alpha,
            "mean_nonconformity_quantum": float(nonconformity_scores.mean()),
            "method": "QCCP",
        }
        
        logger.info(f"  QCCP q_α = {self.q_alpha:.4f}")
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Conformal Prediction — Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate synthetic calibration and test data
    n_cal = 200
    n_test = 500
    
    cal_scores = np.clip(np.random.beta(2, 5, n_cal), 0, 1)
    cal_labels = (cal_scores > 0.3).astype(float) * (np.random.random(n_cal) > 0.2)
    
    test_scores = np.clip(np.random.beta(2, 5, n_test), 0, 1)
    test_labels = (test_scores > 0.3).astype(float) * (np.random.random(n_test) > 0.2)
    
    # Test standard conformal
    predictor = ConformalSepsisPredictor()
    cal_stats = predictor.calibrate(cal_scores, cal_labels)
    print(f"\nCalibration: q_α = {cal_stats['q_alpha']:.4f}")
    
    # Single prediction
    score, lower, upper, coverage = predictor.predict(0.45)
    print(f"Risk=0.45 → [{lower:.3f}, {upper:.3f}] (coverage={coverage:.0%})")
    
    width = upper - lower
    print(f"Width={width:.3f}, escalate={predictor.should_escalate(width)}")
    
    # Batch predictions
    lower_b, upper_b, widths = predictor.predict_batch(test_scores)
    print(f"\nBatch predictions: {len(test_scores)} samples")
    print(f"  Mean width: {widths.mean():.4f}")
    
    # Verify coverage
    coverage_stats = predictor.verify_coverage(test_scores, test_labels)
    
    print("\n✓ Conformal prediction test complete!")
