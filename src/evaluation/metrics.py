"""
QuantumSepsis Shield — Evaluation Metrics
==========================================

Comprehensive metrics for sepsis prediction evaluation:
  - AUROC (Area Under ROC Curve)
  - AUPRC (Area Under Precision-Recall Curve)
  - Sensitivity at 95% specificity
  - Specificity at 95% sensitivity
  - F1 score
  - Lead time analysis
  - Calibration metrics
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: (N,) binary ground truth labels
        y_score: (N,) predicted risk scores [0, 1]
        threshold: Decision threshold for binary predictions
        prefix: Prefix for metric names (e.g., "val_", "test_")
    
    Returns:
        Dictionary of metric_name → metric_value
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    # Sanity checks
    assert len(y_true) == len(y_score), f"Length mismatch: {len(y_true)} vs {len(y_score)}"
    assert y_true.min() >= 0 and y_true.max() <= 1, "y_true must be binary"
    
    metrics = {}
    
    # Handle edge cases
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("Only one class present in y_true. Some metrics unavailable.")
        metrics[f"{prefix}auroc"] = 0.0
        metrics[f"{prefix}auprc"] = 0.0
        return metrics
    
    # 1. AUROC
    metrics[f"{prefix}auroc"] = roc_auc_score(y_true, y_score)
    
    # 2. AUPRC
    metrics[f"{prefix}auprc"] = average_precision_score(y_true, y_score)
    
    # 3. Brier Score (calibration)
    metrics[f"{prefix}brier_score"] = brier_score_loss(y_true, y_score)
    
    # 4. Sensitivity at 95% specificity
    metrics[f"{prefix}sensitivity_at_95spec"] = sensitivity_at_specificity(
        y_true, y_score, target_specificity=0.95
    )
    
    # 5. Specificity at 95% sensitivity
    metrics[f"{prefix}specificity_at_95sens"] = specificity_at_sensitivity(
        y_true, y_score, target_sensitivity=0.95
    )
    
    # 6. Binary predictions at threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics[f"{prefix}accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics[f"{prefix}sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics[f"{prefix}f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    metrics[f"{prefix}true_positives"] = int(tp)
    metrics[f"{prefix}false_positives"] = int(fp)
    metrics[f"{prefix}true_negatives"] = int(tn)
    metrics[f"{prefix}false_negatives"] = int(fn)
    
    # 7. Optimal threshold (Youden's J statistic)
    metrics[f"{prefix}optimal_threshold"] = find_optimal_threshold(y_true, y_score)
    
    return metrics


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Find sensitivity at a given specificity level.
    
    Args:
        y_true: Binary ground truth
        y_score: Predicted scores
        target_specificity: Target specificity level
    
    Returns:
        Sensitivity at the target specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificities = 1 - fpr
    
    # Find the point closest to target specificity (from above)
    valid = specificities >= target_specificity
    if valid.any():
        return tpr[valid][-1]  # Highest sensitivity at or above target specificity
    
    return 0.0


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.95,
) -> float:
    """Find specificity at a given sensitivity level.
    
    Args:
        y_true: Binary ground truth
        y_score: Predicted scores
        target_sensitivity: Target sensitivity level
    
    Returns:
        Specificity at the target sensitivity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificities = 1 - fpr
    
    # Find the point closest to target sensitivity (from above)
    valid = tpr >= target_sensitivity
    if valid.any():
        return specificities[valid][0]  # Highest specificity at or above target sensitivity
    
    return 0.0


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Find optimal threshold using Youden's J statistic.
    
    J = sensitivity + specificity - 1 = TPR - FPR
    
    Returns:
        Optimal decision threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def compute_lead_time_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    hours_before_onset: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute lead time metrics for early detection.
    
    Args:
        y_true: Binary labels
        y_score: Predicted risk scores
        hours_before_onset: Hours before sepsis onset for each window
        threshold: Decision threshold
    
    Returns:
        Dictionary with lead time metrics
    """
    y_pred = (y_score >= threshold).astype(int)
    
    # Focus on sepsis patients
    sepsis_mask = y_true == 1
    if not sepsis_mask.any():
        return {"lead_time_mean": np.nan, "lead_time_median": np.nan}
    
    # True positive lead times
    tp_mask = (y_pred == 1) & (y_true == 1)
    
    if not tp_mask.any():
        return {
            "lead_time_mean": 0.0,
            "lead_time_median": 0.0,
            "lead_time_p25": 0.0,
            "lead_time_p75": 0.0,
            "detection_rate": 0.0,
        }
    
    tp_lead_times = hours_before_onset[tp_mask]
    tp_lead_times = tp_lead_times[~np.isnan(tp_lead_times)]
    
    if len(tp_lead_times) == 0:
        return {
            "lead_time_mean": 0.0,
            "lead_time_median": 0.0,
            "detection_rate": 0.0,
        }
    
    return {
        "lead_time_mean": float(np.mean(tp_lead_times)),
        "lead_time_median": float(np.median(tp_lead_times)),
        "lead_time_p25": float(np.percentile(tp_lead_times, 25)),
        "lead_time_p75": float(np.percentile(tp_lead_times, 75)),
        "lead_time_min": float(np.min(tp_lead_times)),
        "lead_time_max": float(np.max(tp_lead_times)),
        "detection_rate": float(tp_mask.sum() / sepsis_mask.sum()),
    }


def format_metrics(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format metrics dictionary as a readable string."""
    lines = [f"\n{'=' * 50}", f"  {title}", f"{'=' * 50}"]
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            if "threshold" in key or "auroc" in key or "auprc" in key:
                lines.append(f"  {key:35s}: {value:.4f}")
            elif "lead_time" in key:
                lines.append(f"  {key:35s}: {value:.1f} hours")
            else:
                lines.append(f"  {key:35s}: {value:.4f}")
        else:
            lines.append(f"  {key:35s}: {value}")
    
    lines.append(f"{'=' * 50}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.binomial(1, 0.25, n)
    # Generate scores correlated with labels
    y_score = np.clip(
        y_true * 0.6 + np.random.randn(n) * 0.3 + 0.2,
        0, 1
    )
    
    metrics = compute_all_metrics(y_true, y_score, threshold=0.5, prefix="test_")
    print(format_metrics(metrics, "Test Metrics"))
    
    # Lead time test
    hours_before = np.random.uniform(0, 8, n)
    lead_time = compute_lead_time_metrics(y_true, y_score, hours_before)
    print(format_metrics(lead_time, "Lead Time Metrics"))
