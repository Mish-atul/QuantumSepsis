"""
QuantumSepsis Shield — SOFA Score Baseline
============================================

Clinical gold standard baseline: SOFA score threshold.
Represents the current standard-of-care for sepsis detection.

Method: SOFA score ≥ 2 from baseline → sepsis prediction
Expected AUROC: 0.65 – 0.70 (threshold-based, no ML)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class SOFABaseline:
    """SOFA score threshold baseline.
    
    Computes a simplified SOFA-like score from the 12 available features
    and uses a threshold for binary sepsis prediction.
    
    Since we have pre-processed z-normalized features, we approximate
    SOFA component scores using the normalized values:
    
    Cardiovascular:  MAP (index 3) — low MAP → higher SOFA
    Respiratory:     SpO2 (index 6) — low SpO2 → higher SOFA
    Renal:           Creatinine (index 10) — high creatinine → higher SOFA
    Coagulation:     Platelets (index 11) — low platelets → higher SOFA
    Hepatic:         Not available (no bilirubin in our 12 features)
    CNS:             GCS (index 7) — low GCS → higher SOFA
    """
    
    def __init__(self, threshold: float = 2.0):
        """Initialize with SOFA threshold.
        
        Args:
            threshold: SOFA score threshold for prediction (default: 2)
        """
        self.threshold = threshold
    
    def compute_sofa_score(self, window: np.ndarray) -> float:
        """Compute simplified SOFA score from a 6-hour window.
        
        Uses the LATEST values (last time step) of the window.
        
        Args:
            window: (6, 12) feature window (z-normalized or raw)
        
        Returns:
            Simplified SOFA score (0-20 range)
        """
        latest = window[-1]  # Last hour
        
        sofa = 0.0
        
        # Cardiovascular (MAP, index 3): lower = worse
        # In z-normalized space: negative = below mean = concerning
        map_z = latest[3]
        if map_z < -2.0:      sofa += 4
        elif map_z < -1.5:    sofa += 3
        elif map_z < -1.0:    sofa += 2
        elif map_z < -0.5:    sofa += 1
        
        # Respiratory (SpO2, index 6): lower = worse
        spo2_z = latest[6]
        if spo2_z < -2.5:     sofa += 4
        elif spo2_z < -2.0:   sofa += 3
        elif spo2_z < -1.5:   sofa += 2
        elif spo2_z < -1.0:   sofa += 1
        
        # Renal (Creatinine, index 10): higher = worse
        cr_z = latest[10]
        if cr_z > 3.0:        sofa += 4
        elif cr_z > 2.0:      sofa += 3
        elif cr_z > 1.0:      sofa += 2
        elif cr_z > 0.5:      sofa += 1
        
        # Coagulation (Platelets, index 11): lower = worse
        plt_z = latest[11]
        if plt_z < -2.5:      sofa += 4
        elif plt_z < -2.0:    sofa += 3
        elif plt_z < -1.5:    sofa += 2
        elif plt_z < -1.0:    sofa += 1
        
        # CNS (GCS, index 7): lower = worse
        gcs_z = latest[7]
        if gcs_z < -3.0:      sofa += 4
        elif gcs_z < -2.0:    sofa += 3
        elif gcs_z < -1.0:    sofa += 2
        elif gcs_z < -0.5:    sofa += 1
        
        return sofa
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute SOFA scores for a batch of windows.
        
        Args:
            X: (N, 6, 12) batch of windows
        
        Returns:
            (N,) array of SOFA scores
        """
        scores = np.array([self.compute_sofa_score(w) for w in X])
        return scores
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate SOFA baseline on test set.
        
        Args:
            X_test: (N, 6, 12) test windows
            y_test: (N,) labels
            threshold: Optional threshold override
        
        Returns:
            Test metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Compute SOFA scores
        sofa_scores = self.predict_batch(X_test)
        
        # Normalize to [0, 1] for AUROC computation: SOFA max = 20
        sofa_normalized = np.clip(sofa_scores / 20.0, 0, 1)
        
        # Binary predictions at threshold
        metrics = compute_all_metrics(
            y_test, sofa_normalized,
            threshold=threshold / 20.0,
            prefix="sofa_test_"
        )
        
        print(format_metrics(metrics, f"SOFA Baseline (threshold={threshold})"))
        
        # Also sweep thresholds
        best_auroc = metrics.get("sofa_test_auroc", 0)
        logger.info(f"SOFA AUROC: {best_auroc:.4f}")
        
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("SOFA Score Baseline — Synthetic Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create synthetic data with some signal
    n = 1000
    X = np.random.randn(n, 6, 12).astype(np.float32)
    y = (np.random.random(n) > 0.75).astype(int)
    
    # Add some signal: make positive cases have lower MAP, higher creatinine
    for i in range(n):
        if y[i] == 1:
            X[i, :, 3] -= 1.5   # Lower MAP
            X[i, :, 10] += 1.5  # Higher creatinine
            X[i, :, 7] -= 1.0   # Lower GCS
    
    baseline = SOFABaseline(threshold=2.0)
    metrics = baseline.evaluate(X, y)
    
    # Show SOFA distribution
    scores = baseline.predict_batch(X)
    print(f"\nSOFA score distribution:")
    print(f"  Mean:   {scores.mean():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Range:  [{scores.min():.0f}, {scores.max():.0f}]")
    print(f"  Sepsis mean:     {scores[y==1].mean():.1f}")
    print(f"  Non-sepsis mean: {scores[y==0].mean():.1f}")
