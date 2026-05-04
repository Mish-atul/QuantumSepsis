"""
QuantumSepsis Shield — Ensemble Conformal Calibration
======================================================
Runs conformal calibration on ensemble (LSTM + XGBoost) predictions.

Ensemble: 30% LSTM + 70% XGBoost (optimal weights from Phase 1)

Usage:
    python3 scripts/run_ensemble_conformal.py
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_default_config
from src.data.dataset import SepsisDataset
from src.models.lstm import SepsisLSTM
from src.models.conformal import ConformalSepsisPredictor
from src.baselines.xgboost_baseline import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_lstm(checkpoint_path: str, device: torch.device) -> SepsisLSTM:
    """Load LSTM model from checkpoint."""
    config = get_default_config()
    model = SepsisLSTM(config.lstm).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("LSTM loaded | epoch=%s | auroc=%.4f",
                    checkpoint.get("epoch", "?"), checkpoint.get("best_val_auroc", 0))
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_xgboost(checkpoint_path: str):
    """Load XGBoost model from pickle."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    # Handle both dict format and direct model format
    if isinstance(data, dict) and 'model' in data:
        model = data['model']
        logger.info("XGBoost loaded from %s (test AUROC: %.4f)",
                    checkpoint_path, data.get('test_metrics', {}).get('xgb_test_auroc', 0))
    else:
        model = data
        logger.info("XGBoost loaded from %s", checkpoint_path)
    return model


@torch.no_grad()
def run_ensemble_inference(
    lstm_model,
    xgb_model,
    dataset: SepsisDataset,
    batch_size: int,
    device: torch.device,
    split_name: str = "",
    lstm_weight: float = 0.3,
    xgb_weight: float = 0.7,
) -> tuple:
    """Run ensemble inference: 30% LSTM + 70% XGBoost.
    
    Returns:
        (ensemble_scores, labels) as numpy arrays
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    
    all_ensemble_scores = []
    all_labels = []
    n_batches = len(loader)
    
    logger.info("Running ensemble inference on %s split (%d windows, %d batches)...",
                split_name, len(dataset), n_batches)
    
    # Create XGBoostBaseline instance for feature engineering
    xgb_baseline = XGBoostBaseline()
    xgb_baseline.model = xgb_model
    
    t0 = time.time()
    for i, (batch_x, batch_y) in enumerate(loader):
        # LSTM predictions
        batch_x_gpu = batch_x.to(device)
        lstm_out = lstm_model(batch_x_gpu)
        lstm_scores = lstm_out["risk_score"].cpu().numpy()
        
        # XGBoost predictions (needs engineered features)
        batch_x_np = batch_x.numpy()  # (N, 6, 12)
        batch_x_eng = xgb_baseline.engineer_features(batch_x_np)  # (N, 132)
        xgb_proba = xgb_model.predict_proba(batch_x_eng)[:, 1]
        
        # Ensemble: weighted average
        ensemble_scores = lstm_weight * lstm_scores + xgb_weight * xgb_proba
        
        all_ensemble_scores.append(ensemble_scores)
        all_labels.append(batch_y.numpy())
        
        if (i + 1) % max(1, n_batches // 10) == 0:
            logger.info("  %.0f%%  (%d / %d batches)", 100 * (i + 1) / n_batches, i + 1, n_batches)
    
    elapsed = time.time() - t0
    scores = np.concatenate(all_ensemble_scores).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int8)
    
    pos_rate = labels.mean()
    logger.info("  Done in %.1fs | n=%d | pos_rate=%.4f | "
                "score_mean=%.4f | score_std=%.4f | score_range=[%.4f, %.4f]",
                elapsed, len(scores), pos_rate,
                scores.mean(), scores.std(), scores.min(), scores.max())
    
    return scores, labels


def run_ensemble_conformal_calibration(
    lstm_checkpoint: str = "checkpoints/lstm_v1_improved_best.pt",
    xgboost_checkpoint: str = "checkpoints/xgboost_baseline.pkl",
    data_path: str = "data/processed/features.h5",
    output_dir: str = "data/processed",
    batch_size: int = 512,
    device_str: str = "auto",
    lstm_weight: float = 0.3,
    xgb_weight: float = 0.7,
) -> dict:
    """Full ensemble conformal calibration pipeline."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
    
    # Load models
    logger.info("\n%s", "=" * 60)
    logger.info("Step 1/7 — Loading Ensemble Models")
    logger.info("=" * 60)
    lstm_model = load_lstm(lstm_checkpoint, device)
    xgb_model = load_xgboost(xgboost_checkpoint)
    logger.info("Ensemble weights: LSTM=%.1f%%, XGBoost=%.1f%%",
                lstm_weight * 100, xgb_weight * 100)
    
    # Load datasets
    logger.info("\n%s", "=" * 60)
    logger.info("Step 2/7 — Loading val + test splits")
    logger.info("=" * 60)
    
    val_ds = SepsisDataset.from_hdf5(data_path, split="val")
    test_ds = SepsisDataset.from_hdf5(data_path, split="test")
    
    logger.info("Val  | %s", val_ds.summary())
    logger.info("Test | %s", test_ds.summary())
    
    # Inference on val (calibration set)
    logger.info("\n%s", "=" * 60)
    logger.info("Step 3/7 — Ensemble inference on val split")
    logger.info("=" * 60)
    val_scores, val_labels = run_ensemble_inference(
        lstm_model, xgb_model, val_ds, batch_size, device, "val",
        lstm_weight, xgb_weight
    )
    
    # Calibrate
    logger.info("\n%s", "=" * 60)
    logger.info("Step 4/7 — Calibrating ConformalSepsisPredictor")
    logger.info("=" * 60)
    
    config = get_default_config()
    predictor = ConformalSepsisPredictor(config.conformal)
    cal_stats = predictor.calibrate(val_scores, val_labels)
    
    logger.info("Calibration stats:")
    for k, v in cal_stats.items():
        logger.info("  %-30s %s", k, f"{v:.6f}" if isinstance(v, float) else v)
    
    # Inference on test
    logger.info("\n%s", "=" * 60)
    logger.info("Step 5/7 — Ensemble inference on test split")
    logger.info("=" * 60)
    test_scores, test_labels = run_ensemble_inference(
        lstm_model, xgb_model, test_ds, batch_size, device, "test",
        lstm_weight, xgb_weight
    )
    
    # Verify coverage
    logger.info("\n%s", "=" * 60)
    logger.info("Step 6/7 — Verifying empirical coverage")
    logger.info("=" * 60)
    coverage_stats = predictor.verify_coverage(test_scores, test_labels)
    
    logger.info("Coverage verification:")
    for k, v in coverage_stats.items():
        logger.info("  %-30s %.4f", k, v)
    
    empirical = coverage_stats["empirical_coverage"]
    target = coverage_stats["target_coverage"]
    coverage_ok = empirical >= target
    
    if not coverage_ok:
        logger.warning("COVERAGE BELOW GUARANTEE: %.4f < %.4f", empirical, target)
    else:
        logger.info("Coverage OK: %.4f >= %.4f  ✓", empirical, target)
    
    # Generate per-sample intervals
    logger.info("\n%s", "=" * 60)
    logger.info("Step 7/7 — Generating per-sample conformal intervals")
    logger.info("=" * 60)
    
    test_lower, test_upper, test_widths = predictor.predict_batch(test_scores)
    
    escalation_mask = np.array(
        [predictor.should_escalate(w) for w in test_widths], dtype=bool
    )
    n_escalated = int(escalation_mask.sum())
    pct_escalated = 100.0 * n_escalated / len(test_widths)
    
    logger.info("Conformal interval summary (test set):")
    logger.info("  q_alpha (half-width)       : %.4f", predictor.q_alpha)
    logger.info("  Mean interval width        : %.4f", float(test_widths.mean()))
    logger.info("  Median interval width      : %.4f", float(np.median(test_widths)))
    logger.info("  Windows requiring escalation : %d / %d  (%.1f%%)",
                n_escalated, len(test_widths), pct_escalated)
    
    # Save outputs
    intervals_path = output_dir / "ensemble_conformal_test_intervals.npz"
    np.savez(
        intervals_path,
        risk_scores=test_scores,
        lower=test_lower,
        upper=test_upper,
        widths=test_widths,
        labels=test_labels,
        escalation=escalation_mask.astype(np.uint8),
    )
    logger.info("\nSaved test intervals → %s", intervals_path)
    
    val_lower, val_upper, val_widths = predictor.predict_batch(val_scores)
    val_scores_path = output_dir / "ensemble_conformal_val_scores.npz"
    np.savez(
        val_scores_path,
        risk_scores=val_scores,
        lower=val_lower,
        upper=val_upper,
        widths=val_widths,
        labels=val_labels,
    )
    logger.info("Saved val scores     → %s", val_scores_path)
    
    # Save calibration state
    pos_mask = test_labels == 1
    neg_mask = ~pos_mask
    
    calibration_state = {
        "method": "split_conformal_ensemble",
        "ensemble_weights": {"lstm": lstm_weight, "xgboost": xgb_weight},
        "alpha": config.conformal.alpha,
        "coverage_guarantee": 1.0 - config.conformal.alpha,
        "q_alpha": float(predictor.q_alpha),
        "escalation_threshold": config.conformal.escalation_width_threshold,
        "lstm_checkpoint": str(lstm_checkpoint),
        "xgboost_checkpoint": str(xgboost_checkpoint),
        "data_used": str(data_path),
        "calibration_stats": {k: (float(v) if isinstance(v, float) else int(v))
                              for k, v in cal_stats.items()},
        "coverage_verification": {k: float(v) for k, v in coverage_stats.items()},
        "test_interval_summary": {
            "n_test_windows": int(len(test_scores)),
            "n_positive_windows": int(pos_mask.sum()),
            "n_negative_windows": int(neg_mask.sum()),
            "mean_width": float(test_widths.mean()),
            "median_width": float(np.median(test_widths)),
            "min_width": float(test_widths.min()),
            "max_width": float(test_widths.max()),
            "n_escalated": n_escalated,
            "pct_escalated": float(pct_escalated),
            "mean_score_positive": float(test_scores[pos_mask].mean()) if pos_mask.any() else None,
            "mean_score_negative": float(test_scores[neg_mask].mean()) if neg_mask.any() else None,
        },
        "coverage_ok": bool(coverage_ok),
        "files": {
            "intervals_npz": str(intervals_path),
            "val_scores_npz": str(val_scores_path),
        },
    }
    
    json_path = output_dir / "ensemble_conformal_calibration.json"
    with open(json_path, "w") as f:
        json.dump(calibration_state, f, indent=2)
    logger.info("Saved calibration state → %s", json_path)
    
    # Final summary
    logger.info("\n%s", "=" * 60)
    logger.info("ENSEMBLE CONFORMAL CALIBRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Ensemble: %.0f%% LSTM + %.0f%% XGBoost",
                lstm_weight * 100, xgb_weight * 100)
    logger.info("  q_alpha (±half-width)       : %.4f", predictor.q_alpha)
    logger.info("  Target coverage             : %.0f%%", target * 100)
    logger.info("  Empirical coverage (test)   : %.2f%%", empirical * 100)
    logger.info("  Mean interval width (test)  : %.4f", float(test_widths.mean()))
    logger.info("  Windows escalated (test)    : %.1f%%", pct_escalated)
    logger.info("  Coverage guarantee met?     : %s", "YES ✓" if coverage_ok else "NO ✗")
    logger.info("=" * 60)
    
    return calibration_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble conformal calibration")
    parser.add_argument("--lstm-checkpoint", default="checkpoints/lstm_v1_improved_best.pt")
    parser.add_argument("--xgboost-checkpoint", default="checkpoints/xgboost_baseline.pkl")
    parser.add_argument("--data", default="data/processed/features.h5")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lstm-weight", type=float, default=0.3)
    parser.add_argument("--xgb-weight", type=float, default=0.7)
    args = parser.parse_args()
    
    run_ensemble_conformal_calibration(
        lstm_checkpoint=args.lstm_checkpoint,
        xgboost_checkpoint=args.xgboost_checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        lstm_weight=args.lstm_weight,
        xgb_weight=args.xgb_weight,
    )
