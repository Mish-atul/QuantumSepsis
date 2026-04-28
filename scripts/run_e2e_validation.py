"""
QuantumSepsis Shield — End-to-End Orchestrator Validation
==========================================================
Wires: LSTM → Conformal → RedTeamAgent → Orchestrator → Metrics

Usage:
    python3 scripts/run_e2e_validation.py              # real data on GPU server
    python3 scripts/run_e2e_validation.py --synthetic  # local test, no GPU needed

Outputs:
    data/processed/e2e_validation_results.json   ← full metrics + alert distribution
    data/processed/e2e_decisions.npz             ← per-window decisions array
"""

import argparse
import json
import logging
import sys
import tempfile
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ALERT_LEVELS = ["WATCH", "AMBER", "CRITICAL"]  # FAST-TRACK maps to CRITICAL internally


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load infrastructure
# ══════════════════════════════════════════════════════════════════════════════

def load_conformal_predictor(calibration_json_path: str) -> ConformalSepsisPredictor:
    """Reconstruct a calibrated ConformalSepsisPredictor from saved JSON state.

    The conformal calibration script saves q_alpha to JSON.
    We restore it here without re-running calibration.
    """
    path = Path(calibration_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Conformal calibration JSON not found: {path}\n"
            "Run first:\n"
            "  python3 scripts/run_conformal_calibration.py"
        )

    with open(path) as f:
        state = json.load(f)

    config = get_default_config()
    predictor = ConformalSepsisPredictor(config.conformal)
    predictor.q_alpha   = float(state["q_alpha"])
    predictor.calibrated = True

    logger.info("Conformal predictor restored:")
    logger.info("  q_alpha             : %.4f", predictor.q_alpha)
    logger.info("  coverage_guarantee  : %.0f%%", state["coverage_guarantee"] * 100)
    logger.info("  escalation_threshold: %.2f", config.conformal.escalation_width_threshold)
    return predictor


def load_norm_stats(norm_stats_path: str) -> dict:
    """Load normalization stats saved by preprocessing.py.

    Used by RedTeamAgent to denormalize z-scored vitals back to clinical units
    before applying threshold checks.
    """
    path = Path(norm_stats_path)
    if not path.exists():
        logger.warning(
            "normalization_stats.json not found at %s. "
            "RedTeamAgent will run on z-normalized values (thresholds may not apply correctly).",
            path,
        )
        return {}

    with open(path) as f:
        stats = json.load(f)

    logger.info("Normalization stats loaded from %s", path)
    return stats


def load_model(checkpoint_path: str, device: torch.device) -> SepsisLSTM:
    """Load trained SepsisLSTM from checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run LSTM training first:\n"
            "  CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm "
            "--data data/processed/features.h5"
        )

    config = get_default_config()
    model  = SepsisLSTM(config.lstm).to(device)
    ckpt   = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("LSTM loaded | epoch=%s | best_val_auroc=%.4f",
                    ckpt.get("epoch", "?"), ckpt.get("best_val_auroc", float("nan")))
    else:
        model.load_state_dict(ckpt)
        logger.info("LSTM loaded (bare state dict)")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Batched LSTM inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, dataset, batch_size, device, split_name="test"):
    """Batched inference → (risk_scores, raw_windows, labels) all as numpy."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    all_scores, all_windows, all_labels = [], [], []
    n = len(loader)
    logger.info("Inference on %s split (%d windows, %d batches)...", split_name, len(dataset), n)
    t0 = time.time()

    for i, (batch_x, batch_y) in enumerate(loader):
        out = model(batch_x.to(device))
        all_scores.append(out["risk_score"].cpu().numpy())
        all_windows.append(batch_x.numpy())           # keep raw windows for RedTeam
        all_labels.append(batch_y.numpy())
        if (i + 1) % max(1, n // 5) == 0:
            logger.info("  %d/%d batches (%.0f%%)", i + 1, n, 100*(i+1)/n)

    scores  = np.concatenate(all_scores).astype(np.float32)
    windows = np.concatenate(all_windows).astype(np.float32)   # (N, 6, 12)
    labels  = np.concatenate(all_labels).astype(np.int8)

    logger.info("  Done in %.1fs | n=%d | pos_rate=%.4f | "
                "score mean=%.4f std=%.4f range=[%.4f, %.4f]",
                time.time()-t0, len(scores), labels.mean(),
                scores.mean(), scores.std(), scores.min(), scores.max())
    return scores, windows, labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RedTeamAgent evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_red_team(windows: np.ndarray, norm_stats: dict) -> list:
    """Evaluate RedTeamAgent on every test window.

    Args:
        windows   : (N, 6, 12) array of z-normalized vitals windows
        norm_stats: dict with train_mean / train_std for denormalization

    Returns:
        List of N RedTeamAssessment objects
    """
    use_norm = bool(norm_stats)
    agent    = RedTeamAgent(use_normalized=use_norm, norm_stats=norm_stats or None)

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
                time.time()-t0, n_critical, n_amber, n_watch)
    return assessments


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Orchestrator fusion
# ══════════════════════════════════════════════════════════════════════════════

def run_orchestrator(risk_scores, conformal_predictor, red_team_assessments):
    """Fuse risk score + conformal interval + RedTeam → final alert per window.

    Returns:
        decisions   : list of OrchestratorDecision objects
        alert_labels: np.ndarray of int (0=WATCH, 1=AMBER, 2=CRITICAL/FT)
    """
    config       = get_default_config()
    orchestrator = ConfidenceGatedOrchestrator(config.orchestrator)

    logger.info("Running Orchestrator on %d windows...", len(risk_scores))
    t0 = time.time()

    decisions    = []
    alert_labels = np.zeros(len(risk_scores), dtype=np.int8)
    n_ft         = 0

    for i, (score, rt) in enumerate(zip(risk_scores, red_team_assessments)):
        score = float(score)
        _, lower, upper, _ = conformal_predictor.predict(score)

        decision = orchestrator.decide(
            risk_score      = score,
            conformal_lower = lower,
            conformal_upper = upper,
            red_team        = rt,
        )
        decisions.append(decision)

        lvl = decision.alert_level
        if lvl == "WATCH":
            alert_labels[i] = 0
        elif lvl == "AMBER":
            alert_labels[i] = 1
        else:                       # CRITICAL or FAST-TRACK
            alert_labels[i] = 2
            if decision.fast_tracked:
                n_ft += 1

        if (i + 1) % max(1, len(risk_scores) // 5) == 0:
            logger.info("  %d/%d decisions made", i + 1, len(risk_scores))

    logger.info("  Done in %.1fs | FAST-TRACK windows: %d", time.time()-t0, n_ft)
    return decisions, alert_labels, n_ft


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(true_labels, alert_labels, decisions, risk_scores, n_fast_tracked):
    """Compute clinical and ML metrics from orchestrator decisions.

    true_labels  : (N,) binary — 1 = sepsis onset within 4h
    alert_labels : (N,) 0=WATCH / 1=AMBER / 2=CRITICAL
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    N       = len(true_labels)
    pos     = true_labels == 1
    neg     = true_labels == 0

    # --- Alert distribution ---
    n_watch    = int((alert_labels == 0).sum())
    n_amber    = int((alert_labels == 1).sum())
    n_critical = int((alert_labels == 2).sum())

    # --- Treat CRITICAL/FAST-TRACK as "positive" prediction ---
    pred_pos = alert_labels == 2   # CRITICAL = alerted

    tp = int((pred_pos &  pos).sum())
    fp = int((pred_pos & ~pos).sum())
    fn = int((~pred_pos &  pos).sum())
    tn = int((~pred_pos & ~neg).sum()) if neg.any() else 0

    sensitivity = tp / max(tp + fn, 1)       # recall for sepsis class
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)       # precision
    f1          = 2*tp / max(2*tp + fp + fn, 1)

    # False negatives at WATCH level (missed sepsis — most dangerous)
    fn_watch = int(((alert_labels == 0) & pos).sum())

    # AUROC + AUPRC on raw risk scores (continuous metric)
    try:
        auroc = float(roc_auc_score(true_labels, risk_scores))
        auprc = float(average_precision_score(true_labels, risk_scores))
    except Exception:
        auroc, auprc = float("nan"), float("nan")

    # Confidence stats
    widths      = np.array([d.conformal_width for d in decisions], dtype=np.float32)
    confidences = np.array([d.confidence      for d in decisions], dtype=np.float32)
    rt_overrides = int(sum(1 for d in decisions if d.red_team_override != "WATCH"))

    metrics = {
        "n_total"          : N,
        "n_sepsis_positive": int(pos.sum()),
        "n_sepsis_negative": int(neg.sum()),
        "positive_rate"    : float(pos.mean()),

        "alert_distribution": {
            "WATCH"   : n_watch,
            "AMBER"   : n_amber,
            "CRITICAL": n_critical,
            "FAST_TRACK_subset": n_fast_tracked,
            "pct_watch"   : round(100*n_watch/N, 2),
            "pct_amber"   : round(100*n_amber/N, 2),
            "pct_critical": round(100*n_critical/N, 2),
        },

        "clinical_metrics": {
            "sensitivity_at_critical" : round(sensitivity, 4),
            "specificity_at_critical" : round(specificity, 4),
            "ppv_at_critical"         : round(ppv, 4),
            "f1_at_critical"          : round(f1, 4),
            "fn_at_watch"             : fn_watch,
            "pct_sepsis_missed"       : round(100*fn_watch/max(pos.sum(),1), 2),
        },

        "continuous_metrics": {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
        },

        "confidence_stats": {
            "mean_conformal_width" : round(float(widths.mean()), 4),
            "mean_confidence"      : round(float(confidences.mean()), 4),
            "pct_low_confidence"   : round(100*(confidences < 0.5).mean(), 2),
            "pct_high_confidence"  : round(100*(confidences > 0.8).mean(), 2),
        },

        "red_team_stats": {
            "n_overrides"        : rt_overrides,
            "pct_overrides"      : round(100*rt_overrides/N, 2),
        },
    }

    logger.info("\n--- E2E Validation Metrics ---")
    logger.info("Alert distribution:  WATCH=%d  AMBER=%d  CRITICAL=%d  FT=%d",
                n_watch, n_amber, n_critical, n_fast_tracked)
    logger.info("Sensitivity (CRITICAL vs sepsis): %.4f", sensitivity)
    logger.info("Specificity                      : %.4f", specificity)
    logger.info("F1 (CRITICAL as positive pred)   : %.4f", f1)
    logger.info("False negatives at WATCH level   : %d (%.1f%% of sepsis cases)",
                fn_watch, 100*fn_watch/max(pos.sum(),1))
    logger.info("AUROC (risk score, continuous)   : %.4f", auroc)
    logger.info("AUPRC (risk score, continuous)   : %.4f", auprc)
    logger.info("Red Team overrides               : %d (%.1f%%)", rt_overrides, 100*rt_overrides/N)
    logger.info("Mean conformal interval width    : %.4f", widths.mean())

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_e2e_validation(
    checkpoint_path      : str = "checkpoints/lstm_best.pt",
    data_path            : str = "data/processed/features.h5",
    calibration_json_path: str = "data/processed/conformal_calibration.json",
    norm_stats_path      : str = "data/processed/normalization_stats.json",
    output_dir           : str = "data/processed",
    batch_size           : int = 512,
    device_str           : str = "auto",
) -> dict:
    """Run the full end-to-end orchestrator validation pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)

    # ── Step 1: Load infrastructure ───────────────────────────────────────────
    logger.info("\n%s\nStep 1/5 — Loading infrastructure\n%s", "="*60, "="*60)
    conformal_predictor = load_conformal_predictor(calibration_json_path)
    norm_stats          = load_norm_stats(norm_stats_path)
    model               = load_model(checkpoint_path, device)

    # ── Step 2: Load test dataset + run LSTM inference ────────────────────────
    logger.info("\n%s\nStep 2/5 — LSTM inference on test set\n%s", "="*60, "="*60)

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"HDF5 not found: {data_path}\n"
            "Run windowing first:\n  python3 scripts/run_windowing_real.py"
        )

    test_ds = SepsisDataset.from_hdf5(data_path, split="test")
    logger.info("Test | %s", test_ds.summary())

    risk_scores, windows, true_labels = run_inference(
        model, test_ds, batch_size, device, split_name="test"
    )

    # ── Step 3: RedTeam evaluation ────────────────────────────────────────────
    logger.info("\n%s\nStep 3/5 — RedTeamAgent evaluation\n%s", "="*60, "="*60)
    red_team_assessments = run_red_team(windows, norm_stats)

    # ── Step 4: Orchestrator fusion ───────────────────────────────────────────
    logger.info("\n%s\nStep 4/5 — Orchestrator fusion\n%s", "="*60, "="*60)
    decisions, alert_labels, n_fast_tracked = run_orchestrator(
        risk_scores, conformal_predictor, red_team_assessments
    )

    # ── Step 5: Metrics ───────────────────────────────────────────────────────
    logger.info("\n%s\nStep 5/5 — Computing metrics\n%s", "="*60, "="*60)
    metrics = compute_metrics(
        true_labels, alert_labels, decisions, risk_scores, n_fast_tracked
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "pipeline": "e2e_validation",
        "checkpoint"      : str(checkpoint_path),
        "data"            : str(data_path),
        "calibration_json": str(calibration_json_path),
        "q_alpha_used"    : float(conformal_predictor.q_alpha),
        **metrics,
    }

    json_path = output_dir / "e2e_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved results → %s", json_path)

    # Save per-window decisions array
    npz_path = output_dir / "e2e_decisions.npz"
    np.savez(
        npz_path,
        risk_scores  = risk_scores,
        alert_labels = alert_labels,       # 0=WATCH 1=AMBER 2=CRITICAL
        true_labels  = true_labels,
        conformal_widths = np.array([d.conformal_width for d in decisions], dtype=np.float32),
        confidences      = np.array([d.confidence      for d in decisions], dtype=np.float32),
        fast_tracked     = np.array([d.fast_tracked     for d in decisions], dtype=np.uint8),
        red_team_levels  = np.array(
            [{"WATCH":0,"AMBER":1,"CRITICAL":2}.get(d.red_team_override, 0)
             for d in decisions], dtype=np.int8
        ),
    )
    logger.info("Saved per-window decisions → %s", npz_path)

    logger.info("\n%s\nE2E VALIDATION COMPLETE\n%s", "="*60, "="*60)
    logger.info("Sensitivity (CRITICAL alert vs sepsis): %.4f",
                metrics["clinical_metrics"]["sensitivity_at_critical"])
    logger.info("Sepsis cases missed at WATCH           : %d (%.1f%%)",
                metrics["clinical_metrics"]["fn_at_watch"],
                metrics["clinical_metrics"]["pct_sepsis_missed"])
    logger.info("AUROC                                  : %.4f",
                metrics["continuous_metrics"]["auroc"])
    logger.info("%s", "="*60)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test(output_dir: str = "data/processed") -> None:
    """Full e2e smoke test — no GPU or MIMIC-IV data required."""
    import h5py
    logger.info("="*60)
    logger.info("SYNTHETIC SMOKE TEST — E2E Validation")
    logger.info("="*60)

    config = get_default_config()
    rng    = np.random.default_rng(42)

    def make_split(n, pos_rate=0.13):
        labels = (rng.random(n) < pos_rate).astype(np.float32)
        X = rng.standard_normal((n, 6, 12)).astype(np.float32)
        X[labels == 1] += 0.3
        return X, labels.astype(np.int8)

    X_train, y_train = make_split(500)
    X_val,   y_val   = make_split(200)
    X_test,  y_test  = make_split(300)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # HDF5
        h5_path = tmpdir / "features.h5"
        with h5py.File(h5_path, "w") as f:
            for split, X, y in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
                f.create_dataset(f"X_{split}", data=X)
                f.create_dataset(f"y_{split}", data=y)

        # Checkpoint
        model = SepsisLSTM(config.lstm)
        ckpt_path = tmpdir / "lstm_best.pt"
        torch.save({"epoch":0,"model_state_dict":model.state_dict(),
                    "best_val_auroc":0.5,"metrics":{}}, ckpt_path)

        # Conformal calibration JSON (simulate what run_conformal_calibration.py outputs)
        cal_json = tmpdir / "conformal_calibration.json"
        with open(cal_json, "w") as f:
            json.dump({
                "q_alpha": 0.25,
                "coverage_guarantee": 0.90,
                "method": "split_conformal",
            }, f)

        # No norm stats (RedTeam runs on z-normalized values, warns but continues)
        result = run_e2e_validation(
            checkpoint_path       = str(ckpt_path),
            data_path             = str(h5_path),
            calibration_json_path = str(cal_json),
            norm_stats_path       = str(tmpdir / "does_not_exist.json"),
            output_dir            = str(tmpdir / "out"),
            batch_size            = 32,
            device_str            = "cpu",
        )

        assert (tmpdir / "out" / "e2e_validation_results.json").exists()
        assert (tmpdir / "out" / "e2e_decisions.npz").exists()
        assert "clinical_metrics" in result
        assert "alert_distribution" in result
        assert 0 <= result["clinical_metrics"]["sensitivity_at_critical"] <= 1

        logger.info("\n✓ Synthetic smoke test passed!")
        logger.info("  Alert distribution: %s", result["alert_distribution"])
        logger.info("  Sensitivity: %.4f", result["clinical_metrics"]["sensitivity_at_critical"])
        logger.info("  Sepsis missed (WATCH): %d", result["clinical_metrics"]["fn_at_watch"])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end orchestrator validation")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_best.pt")
    parser.add_argument("--data",       default="data/processed/features.h5")
    parser.add_argument("--calibration-json",
                        default="data/processed/conformal_calibration.json")
    parser.add_argument("--norm-stats",
                        default="data/processed/normalization_stats.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Run smoke test without real data")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_test(output_dir=args.output_dir)
    else:
        run_e2e_validation(
            checkpoint_path       = args.checkpoint,
            data_path             = args.data,
            calibration_json_path = args.calibration_json,
            norm_stats_path       = args.norm_stats,
            output_dir            = args.output_dir,
            batch_size            = args.batch_size,
            device_str            = args.device,
        )
