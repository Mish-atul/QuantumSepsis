"""
Aggregate window-level predictions to stay-level and compute AUROC.
Standard methodology for sepsis prediction papers.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("STAY-LEVEL METRICS COMPUTATION")
    logger.info("="*60)
    
    # Load window-level predictions from e2e validation
    logger.info("Loading window-level predictions...")
    decisions = np.load('data/processed/e2e_decisions.npz')
    risk_scores = decisions['risk_scores']
    alert_labels = decisions['alert_labels']
    
    # Load true labels
    with h5py.File('data/processed/features.h5', 'r') as f:
        y_test = f['y_test'][:]
    
    logger.info(f"Loaded {len(risk_scores)} window-level predictions")
    
    # Load test features to get stay_id mapping
    logger.info("Loading test features for stay_id mapping...")
    test_df = pd.read_parquet('data/processed/test_features.parquet')
    
    # Create stay_id mapping
    # Each row in test_df corresponds to an hour, and windowing creates overlapping windows
    # We need to map each window back to its stay_id
    
    # Simplified approach: assume windows are created in order from test_df
    # Each stay contributes multiple windows
    logger.info("Mapping windows to stays...")
    
    # Get unique stay_ids from test_df
    stay_ids_in_test = test_df['stay_id'].unique()
    logger.info(f"Found {len(stay_ids_in_test)} unique stays in test set")
    
    # For each stay, find all its windows
    # This requires knowing the windowing logic: 6-hour windows with 1-hour stride
    stay_scores = defaultdict(list)
    stay_labels = {}
    
    # Group test_df by stay_id
    for stay_id, group in test_df.groupby('stay_id'):
        # Each stay has N hours of data
        n_hours = len(group)
        
        # Number of windows for this stay: max(0, n_hours - window_size + 1)
        # with window_size=6 and stride=1
        n_windows = max(0, n_hours - 6 + 1)
        
        if n_windows > 0:
            # Find the window indices for this stay
            # This is approximate - we'd need the exact windowing metadata
            # For now, use a heuristic based on cumulative window counts
            pass
    
    # Alternative approach: use the fact that windows are sequential
    # and test_df is ordered by stay_id and time
    logger.info("Using sequential window assignment...")
    
    # Create a mapping from window index to stay_id
    window_to_stay = []
    current_window_idx = 0
    
    for stay_id, group in test_df.groupby('stay_id', sort=False):
        n_hours = len(group)
        n_windows = max(0, n_hours - 6 + 1)
        
        for _ in range(n_windows):
            window_to_stay.append(stay_id)
        
        current_window_idx += n_windows
    
    logger.info(f"Mapped {len(window_to_stay)} windows to stays")
    
    # Verify we have the right number of windows
    if len(window_to_stay) != len(risk_scores):
        logger.warning(f"Window count mismatch: {len(window_to_stay)} mapped vs {len(risk_scores)} actual")
        logger.warning("Using truncated mapping...")
        min_len = min(len(window_to_stay), len(risk_scores))
        window_to_stay = window_to_stay[:min_len]
        risk_scores = risk_scores[:min_len]
        y_test = y_test[:min_len]
        alert_labels = alert_labels[:min_len]
    
    # Aggregate by stay: max risk score
    logger.info("Aggregating windows to stay-level...")
    for i, stay_id in enumerate(window_to_stay):
        stay_scores[stay_id].append(risk_scores[i])
        # Stay is positive if ANY window is positive
        stay_labels[stay_id] = max(stay_labels.get(stay_id, 0), y_test[i])
    
    # Convert to arrays
    stay_ids = list(stay_scores.keys())
    y_stay = np.array([stay_labels[sid] for sid in stay_ids])
    scores_stay = np.array([max(stay_scores[sid]) for sid in stay_ids])
    
    logger.info(f"Aggregated to {len(stay_ids)} stays")
    logger.info(f"Sepsis stays: {y_stay.sum()} ({100*y_stay.mean():.2f}%)")
    
    # Compute stay-level metrics
    logger.info("Computing stay-level metrics...")
    auroc = roc_auc_score(y_stay, scores_stay)
    auprc = average_precision_score(y_stay, scores_stay)
    
    # Compute confusion matrix at optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_stay, scores_stay)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (scores_stay >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_stay, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    results = {
        'n_stays': int(len(stay_ids)),
        'n_sepsis_stays': int(y_stay.sum()),
        'sepsis_prevalence': float(y_stay.mean()),
        'stay_level_auroc': float(auroc),
        'stay_level_auprc': float(auprc),
        'optimal_threshold': float(optimal_threshold),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }
    
    logger.info("="*60)
    logger.info("STAY-LEVEL METRICS")
    logger.info("="*60)
    logger.info(f"Number of stays:        {results['n_stays']}")
    logger.info(f"Sepsis stays:           {results['n_sepsis_stays']} ({100*results['sepsis_prevalence']:.2f}%)")
    logger.info(f"Stay-level AUROC:       {results['stay_level_auroc']:.4f}")
    logger.info(f"Stay-level AUPRC:       {results['stay_level_auprc']:.4f}")
    logger.info(f"Optimal threshold:      {results['optimal_threshold']:.4f}")
    logger.info(f"Sensitivity:            {results['sensitivity']:.4f}")
    logger.info(f"Specificity:            {results['specificity']:.4f}")
    logger.info(f"PPV:                    {results['ppv']:.4f}")
    logger.info(f"NPV:                    {results['npv']:.4f}")
    logger.info("="*60)
    
    # Save results
    output_path = Path('data/processed/stay_level_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved → {output_path}")
    
    return results


if __name__ == '__main__':
    main()
