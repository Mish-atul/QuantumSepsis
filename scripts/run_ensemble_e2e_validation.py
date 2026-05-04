"""
QuantumSepsis Shield — Ensemble E2E Validation
===============================================
Full pipeline: Ensemble → Conformal → RedTeam → Orchestrator

Usage:
    python3 scripts/run_ensemble_e2e_validation.py
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
from src.agents.red_team import RedTeamAgent
from src.agents.orchestrator import ConfidenceGatedOrchestrator
from src.baselines.xgboost_baseline import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_lstm(checkpoint_path: str, device: torch.device):
    """Load LSTM model."""
    config = get_default_config()
    model = SepsisLSTM(config.lstm).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_xgboost(checkpoint_path: str):
    """Load XGBoost model."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    # Handle both dict format and direct model format
    if isinstance(data, dict) and 'model' in data:
        return data['model']
    return data


def load_conformal_predictor(calibration_json_path: str) -> ConformalSepsisPredictor:
    """Load calibrated conformal predictor from JSON."""
    with open(calibration_json_path) as f:
        state = json.load(f)
    
    config = get_default_config()
    predictor = ConformalSepsisPredictor(config.conformal)
    predictor.q_alpha = float(state["q_alpha"])
    predictor.calibrated = True
    
    logger.info("Conformal predictor restored:")
    logger.info("  q_alpha             : %.4f", predictor.q_alpha)
    logger.info("  coverage_guarantee  : %.0f%%", state["coverage_guarantee"] * 100)
    return predictor


def load_norm_stats(norm_stats_path: str) -> dict:
    """Load normalization stats for RedTeam denormalization."""
    path = Path(norm_stats_path)
    if not path.exists():
        logger.warning("normalization_stats.json not found. RedTeam will use z-normalized values.")
        return {}
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def run_ensemble_inference(
    lstm_model,
    xgb_model,
    dataset,
    batch_size,
    device,
    split_name="test",
    lstm_weight=0.3,
    xgb_weight=0.7,
):
    """Run ensemble inference and return (scores, windows, labels)."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    
    all_scores, all_windows, all_labels = [], [], []
    n = len(loader)
    logger.info("Ensemble inference on %s split (%d windows, %d batches)...",
                split_name, len(dataset), n)
    
    # Create XGBoostBaseline instance for feature engineering
    xgb_baseline = XGBoostBaseline()
    xgb_baseline.model = xgb_model
    
    t0 = time.time()
    
    for i, (batch_x, batch_y) in enumerate(loader):
        # LSTM
        lstm_out = lstm_model(batch_x.to(device))
        lstm_scores = lstm_out["risk_score"].cpu().numpy()
        
        # XGBoost (needs engineered features)
        batch_x_np = batch_x.numpy()  # (N, 6, 12)
        batch_x_eng = xgb_baseline.engineer_features(batch_x_np)  # (N, 132)
        xgb_proba = xgb_model.predict_proba(batch_x_eng)[:, 1]
        
        # Ensemble
        ensemble_scores = lstm_weight * lstm_scores + xgb_weight * xgb_proba
        
        all_scores.append(ensemble_scores)
        all_windows.append(batch_x_np)
        all_labels.append(batch_y.numpy())
        
        if (i + 1) % max(1, n // 5) == 0:
            logger.info("  %d/%d batches (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    
    scores = np.concatenate(all_scores).astype(np.float32)
    windows = np.concatenate(all_windows).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int8)
    
    logger.info("  Done in %.1fs | n=%d | pos_rate=%.4f | "
                "score mean=%.4f std=%.4f range=[%.4f, %.4f]",
                time.time() - t0, len(scores), labels.mean(),
                scores.mean(), scores.std(), scores.min(), scores.max())
    
    return scores, windows, labels


def run_red_team(windows: np.ndarray, norm_stats: dict) -> list:
    """Run RedTeamAgent on all windows."""
    use_norm = bool(norm_stats)
    agent = RedTeamAgent(use_normalized=use_norm, norm_stats=norm_stats or None)
    
    logger.info("Running RedTeamAgent on %d windows (denormalize=%s)...",
                len(windows), use_norm)
    t0 = time.time()
    
    assessments = []
    n_critical, n_amber, n_watch = 0, 0, 0
    
    for i, window in enumerate(windows):
        a = agent.evaluate(window)
        assessments.append(a)
        if a.override_level == "CRITICAL":
            n_critical += 1
        elif a.override_level == "AMBER":
            n_amber += 1
        else:
            n_watch += 1
        
        if (i + 1) % max(1, len(windows) // 5) == 0:
            logger.info("  %d/%d windows processed", i + 1, len(windows))
    
    logger.info("  Done in %.1fs | CRITICAL=%d  AMBER=%d  WATCH=%d",
                time.time() - t0, n_critical, n_amber, n_watch)
    return assessments


def run_orchestrator(risk_scores, conformal_predictor, red_team_assessments):
    """Run orchestrator fusion."""
    config = get_default_config()
    orchestrator = ConfidenceGatedOrchestrator(config.orchestrator)
    
    logger.info("Running Orchestrator on %d windows...", len(risk_scores))
    t0 = time.time()
    
    decisions = []
    alert_labels = np.zeros(len(risk_scores), dtype=np.int8)
    n_ft = 0
    
    for i, (score, rt) in enumerate(zip(risk_scores, red_team_assessments)):
        score = float(score)
        _, lower, upper, _ = conformal_predictor.predict(score)
        
        decision = orchestrator.decide(
            risk_score=score,
            conformal_lower=lower,
            conformal_upper=upper,
            red_team=rt,
        )
        decisions.append(decision)
        
        lvl = decision.alert_level
        if lvl == "WATCH":
            alert_labels[i] = 0
        elif lvl == "AMBER":
            alert_labels[i] = 1
        else:
            alert_labels[i] = 2
            if decision.fast_tracked:
                n_ft += 1
        
        if (i + 1) % max(1, len(risk_scores) // 5) == 0:
            logger.info("  %d/%d decisions made", i + 1, len(risk_scores))
    
    logger.info("  Done in %.1fs | FAST-TRACK windows: %d", time.time() - t0, n_ft)
    return decisions, alert_labels, n_ft


def compute_metrics(true_labels, alert_labels, decisions, risk_scores, n_fast_tracked):
    """Compute clinical and ML metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    N = len(true_labels)
    pos = true_labels == 1
    neg = true_labels == 0
    
    n_watch = int((alert_labels == 0).sum())
    n_amber = int((alert_labels == 1).sum())
    n_critical = int((alert_labels == 2).sum())
    
    pred_pos = alert_labels == 2
    
    tp = int((pred_pos & pos).sum())
    fp = int((pred_pos & ~pos).sum())
    fn = int((~pred_pos & pos).sum())
    tn = int((~pred_pos & ~neg).sum()) if neg.any() else 0
    
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    
    fn_watch = int(((alert_labels == 0) & pos).sum())
    
    try:
        auroc = float(roc_auc_score(true_labels, risk_scores))
        auprc = float(average_precision_score(true_labels, risk_scores))
    except Exception:
        auroc, auprc = float("nan"), float("nan")
    
    widths = np.array([d.conformal_width for d in decisions], dtype=np.float32)
    confidences = np.array([d.confidence for d in decisions], dtype=np.float32)
    rt_overrides = int(sum(1 for d in decisions if d.red_team_override != "WATCH"))
    
    metrics = {
        "n_total": N,
        "n_sepsis_positive": int(pos.sum()),
        "n_sepsis_negative": int(neg.sum()),
        "positive_rate": float(pos.mean()),
        
        "alert_distribution": {
            "WATCH": n_watch,
            "AMBER": n_amber,
            "CRITICAL": n_critical,
            "FAST_TRACK_subset": n_fast_tracked,
            "pct_watch": round(100 * n_watch / N, 2),
            "pct_amber": round(100 * n_amber / N, 2),
            "pct_critical": round(100 * n_critical / N, 2),
        },
        
        "clinical_metrics": {
            "sensitivity_at_critical": round(sensitivity, 4),
            "specificity_at_critical": round(specificity, 4),
            "ppv_at_critical": round(ppv, 4),
            "f1_at_critical": round(f1, 4),
            "fn_at_watch": fn_watch,
            "pct_sepsis_missed": round(100 * fn_watch / max(pos.sum(), 1), 2),
        },
        
        "continuous_metrics": {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        },
        
        "confidence_stats": {
            "mean_conformal_width": round(float(widths.mean()), 4),
            "mean_confidence": round(float(confidences.mean()), 4),
            "pct_low_confidence": round(100 * (confidences < 0.5).mean(), 2),
            "pct_high_confidence": round(100 * (confidences > 0.8).mean(), 2),
        },
        
        "red_team_stats": {
            "n_overrides": rt_overrides,
            "pct_overrides": round(100 * rt_overrides / N, 2),
        },
    }
    
    logger.info("\n--- E2E Validation Metrics ---")
    logger.info("Alert distribution:  WATCH=%d  AMBER=%d  CRITICAL=%d  FT=%d",
                n_watch, n_amber, n_critical, n_fast_tracked)
    logger.info("Sensitivity (CRITICAL vs sepsis): %.4f", sensitivity)
    logger.info("Specificity                      : %.4f", specificity)
    logger.info("F1 (CRITICAL as positive pred)   : %.4f", f1)
    logger.info("False negatives at WATCH level   : %d (%.1f%% of sepsis cases)",
                fn_watch, 100 * fn_watch / max(pos.sum(), 1))
    logger.info("AUROC (risk score, continuous)   : %.4f", auroc)
    logger.info("AUPRC (risk score, continuous)   : %.4f", auprc)
    logger.info("Red Team overrides               : %d (%.1f%%)", rt_overrides, 100 * rt_overrides / N)
    logger.info("Mean conformal interval width    : %.4f", widths.mean())
    
    return metrics


def run_ensemble_e2e_validation(
    lstm_checkpoint: str,
    xgboost_checkpoint: str,
    data_path: str,
    calibration_json_path: str,
    norm_stats_path: str,
    output_dir: str,
    batch_size: int = 512,
    device_str: str = "auto",
    lstm_weight: float = 0.3,
    xgb_weight: float = 0.7,
) -> dict:
    """Full ensemble E2E validation pipeline."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
    
    # Step 1: Load infrastructure
    logger.info("\n%s\nStep 1/5 — Loading infrastructure\n%s", "=" * 60, "=" * 60)
    conformal_predictor = load_conformal_predictor(calibration_json_path)
    norm_stats = load_norm_stats(norm_stats_path)
    lstm_model = load_lstm(lstm_checkpoint, device)
    xgb_model = load_xgboost(xgboost_checkpoint)
    logger.info("Ensemble: %.0f%% LSTM + %.0f%% XGBoost", lstm_weight * 100, xgb_weight * 100)
    
    # Step 2: Load test dataset + run ensemble inference
    logger.info("\n%s\nStep 2/5 — Ensemble inference on test set\n%s", "=" * 60, "=" * 60)
    test_ds = SepsisDataset.from_hdf5(data_path, split="test")
    logger.info("Test | %s", test_ds.summary())
    
    risk_scores, windows, true_labels = run_ensemble_inference(
        lstm_model, xgb_model, test_ds, batch_size, device, "test",
        lstm_weight, xgb_weight
    )
    
    # Step 3: RedTeam evaluation
    logger.info("\n%s\nStep 3/5 — RedTeamAgent evaluation\n%s", "=" * 60, "=" * 60)
    red_team_assessments = run_red_team(windows, norm_stats)
    
    # Step 4: Orchestrator fusion
    logger.info("\n%s\nStep 4/5 — Orchestrator fusion\n%s", "=" * 60, "=" * 60)
    decisions, alert_labels, n_fast_tracked = run_orchestrator(
        risk_scores, conformal_predictor, red_team_assessments
    )
    
    # Step 5: Metrics
    logger.info("\n%s\nStep 5/5 — Computing metrics\n%s", "=" * 60, "=" * 60)
    metrics = compute_metrics(
        true_labels, alert_labels, decisions, risk_scores, n_fast_tracked
    )
    
    # Save results
    results = {
        "pipeline": "ensemble_e2e_validation",
        "ensemble_weights": {"lstm": lstm_weight, "xgboost": xgb_weight},
        "lstm_checkpoint": str(lstm_checkpoint),
        "xgboost_checkpoint": str(xgboost_checkpoint),
        "data": str(data_path),
        "calibration_json": str(calibration_json_path),
        "q_alpha_used": float(conformal_predictor.q_alpha),
        **metrics,
    }
    
    json_path = output_dir / "ensemble_e2e_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved results → %s", json_path)
    
    # Save per-window decisions
    npz_path = output_dir / "ensemble_e2e_decisions.npz"
    np.savez(
        npz_path,
        risk_scores=risk_scores,
        alert_labels=alert_labels,
        true_labels=true_labels,
        conformal_widths=np.array([d.conformal_width for d in decisions], dtype=np.float32),
        confidences=np.array([d.confidence for d in decisions], dtype=np.float32),
        fast_tracked=np.array([d.fast_tracked for d in decisions], dtype=np.uint8),
        red_team_levels=np.array(
            [{"WATCH": 0, "AMBER": 1, "CRITICAL": 2}.get(d.red_team_override, 0)
             for d in decisions], dtype=np.int8
        ),
    )
    logger.info("Saved per-window decisions → %s", npz_path)
    
    logger.info("\n%s\nENSEMBLE E2E VALIDATION COMPLETE\n%s", "=" * 60, "=" * 60)
    logger.info("Sensitivity (CRITICAL alert vs sepsis): %.4f",
                metrics["clinical_metrics"]["sensitivity_at_critical"])
    logger.info("Sepsis cases missed at WATCH           : %d (%.1f%%)",
                metrics["clinical_metrics"]["fn_at_watch"],
                metrics["clinical_metrics"]["pct_sepsis_missed"])
    logger.info("AUROC                                  : %.4f",
                metrics["continuous_metrics"]["auroc"])
    logger.info("%s", "=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble E2E validation")
    parser.add_argument("--lstm-checkpoint", default="checkpoints/lstm_v1_improved_best.pt")
    parser.add_argument("--xgboost-checkpoint", default="checkpoints/xgboost_baseline.pkl")
    parser.add_argument("--data", default="data/processed/features.h5")
    parser.add_argument("--calibration-json",
                        default="data/processed/ensemble_conformal_calibration.json")
    parser.add_argument("--norm-stats",
                        default="data/processed/normalization_stats.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lstm-weight", type=float, default=0.3)
    parser.add_argument("--xgb-weight", type=float, default=0.7)
    args = parser.parse_args()
    
    run_ensemble_e2e_validation(
        lstm_checkpoint=args.lstm_checkpoint,
        xgboost_checkpoint=args.xgboost_checkpoint,
        data_path=args.data,
        calibration_json_path=args.calibration_json,
        norm_stats_path=args.norm_stats,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        lstm_weight=args.lstm_weight,
        xgb_weight=args.xgb_weight,
    )
