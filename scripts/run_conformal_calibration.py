"""
QuantumSepsis Shield — Conformal Calibration Script
=====================================================

Runs split conformal calibration on trained LSTM risk scores.

Pipeline:
  1. Load trained LSTM from checkpoints/lstm_best.pt
  2. Run batched inference on val split  → calibration scores
  3. Calibrate ConformalSepsisPredictor  → q_alpha threshold
  4. Run batched inference on test split → test scores
  5. Verify empirical coverage on test set (must be >= 90%)
  6. Save per-sample conformal intervals for all test windows
  7. Save calibration state to JSON (q_alpha, stats, coverage)

Outputs:
  data/processed/conformal_calibration.json  ← calibration state + stats
  data/processed/conformal_test_intervals.npz ← per-sample [lower, upper, width]
  data/processed/conformal_val_scores.npz    ← val risk scores + labels

Usage:
  # On GPU server with real data:
  python3 scripts/run_conformal_calibration.py

  # Test locally without GPU / real data:
  python3 scripts/run_conformal_calibration.py --synthetic

  # Custom paths:
  python3 scripts/run_conformal_calibration.py \\
      --checkpoint checkpoints/lstm_best.pt \\
      --data data/processed/features.h5 \\
      --output-dir data/processed

Run on GPU server:
  screen -S conformal
  cd ~/QuantumSepsis
  python3 scripts/run_conformal_calibration.py
  # Ctrl+A, D to detach
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_default_config
from src.data.dataset import SepsisDataset
from src.models.lstm import SepsisLSTM
from src.models.conformal import ConformalSepsisPredictor

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: load LSTM from checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> SepsisLSTM:
    """Load SepsisLSTM from a training checkpoint.

    The checkpoint saved by LSTMTrainer._save_checkpoint contains:
        {epoch, model_state_dict, optimizer_state_dict, best_val_auroc, metrics, config}
    """
    config = get_default_config()
    model = SepsisLSTM(config.lstm).to(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run LSTM training first:\n"
            "  CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm "
            "--data data/processed/features.h5"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint stores model_state_dict; handle both formats gracefully
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_epoch = checkpoint.get("epoch", "unknown")
        saved_auroc = checkpoint.get("best_val_auroc", float("nan"))
        logger.info(
            "Loaded checkpoint  epoch=%s  best_val_auroc=%.4f",
            saved_epoch, saved_auroc,
        )
    else:
        # Bare state dict (fallback)
        model.load_state_dict(checkpoint)
        logger.info("Loaded bare state dict from %s", checkpoint_path)

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: batched inference → risk scores
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: SepsisLSTM,
    dataset: SepsisDataset,
    batch_size: int,
    device: torch.device,
    split_name: str = "",
) -> tuple:
    """Run batched model inference and return (risk_scores, labels) as numpy arrays.

    Uses the same batch loop pattern as LSTMTrainer._validate so behaviour
    is identical to what was used during training.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    all_scores = []
    all_labels = []
    n_batches   = len(loader)

    logger.info(
        "Running inference on %s split  (%d windows, %d batches) ...",
        split_name, len(dataset), n_batches,
    )

    t0 = time.time()
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        out = model(batch_x)
        all_scores.append(out["risk_score"].cpu().numpy())
        all_labels.append(batch_y.numpy())

        if (i + 1) % max(1, n_batches // 10) == 0:
            pct = 100 * (i + 1) / n_batches
            logger.info("  %.0f%%  (%d / %d batches)", pct, i + 1, n_batches)

    elapsed = time.time() - t0
    scores = np.concatenate(all_scores).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int8)

    pos_rate = labels.mean()
    logger.info(
        "  Done in %.1fs | n=%d | pos_rate=%.4f | "
        "score_mean=%.4f | score_std=%.4f | score_range=[%.4f, %.4f]",
        elapsed, len(scores), pos_rate,
        scores.mean(), scores.std(), scores.min(), scores.max(),
    )
    return scores, labels


# ══════════════════════════════════════════════════════════════════════════════
#  Core pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_conformal_calibration(
    checkpoint_path: str = "checkpoints/lstm_best.pt",
    data_path: str       = "data/processed/features.h5",
    output_dir: str      = "data/processed",
    batch_size: int      = 512,
    device_str: str      = "auto",
) -> dict:
    """Full conformal calibration pipeline.

    Steps
    -----
    1. Resolve device (CUDA if available).
    2. Load LSTM from checkpoint.
    3. Load val + test splits from HDF5.
    4. Run inference on val → calibration scores.
    5. Calibrate ConformalSepsisPredictor → q_alpha.
    6. Run inference on test → test scores.
    7. Verify empirical coverage on test set.
    8. Produce per-sample conformal intervals for all test windows.
    9. Save everything to disk.

    Returns
    -------
    dict with calibration stats + coverage verification results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. device ──────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # ── 2. load model ──────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 1/7 — Loading LSTM model")
    logger.info("=" * 60)
    model = load_model(checkpoint_path, device)
    logger.info(model.summary())

    # ── 3. load datasets ───────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 2/7 — Loading val + test splits from HDF5")
    logger.info("=" * 60)

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"HDF5 not found: {data_path}\n"
            "Run windowing first:\n"
            "  python3 scripts/run_windowing_real.py"
        )

    val_ds  = SepsisDataset.from_hdf5(data_path, split="val")
    test_ds = SepsisDataset.from_hdf5(data_path, split="test")

    logger.info("Val  | %s", val_ds.summary())
    logger.info("Test | %s", test_ds.summary())

    # ── 4. inference on val (calibration set) ─────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 3/7 — Inference on val split (calibration set)")
    logger.info("=" * 60)
    val_scores, val_labels = run_inference(
        model, val_ds, batch_size, device, split_name="val",
    )

    # ── 5. calibrate ──────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 4/7 — Calibrating ConformalSepsisPredictor")
    logger.info("=" * 60)

    config    = get_default_config()
    predictor = ConformalSepsisPredictor(config.conformal)
    cal_stats = predictor.calibrate(val_scores, val_labels)

    logger.info("Calibration stats:")
    for k, v in cal_stats.items():
        logger.info("  %-30s %s", k, f"{v:.6f}" if isinstance(v, float) else v)

    # ── 6. inference on test ──────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 5/7 — Inference on test split")
    logger.info("=" * 60)
    test_scores, test_labels = run_inference(
        model, test_ds, batch_size, device, split_name="test",
    )

    # ── 7. verify coverage ────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 6/7 — Verifying empirical coverage on test set")
    logger.info("=" * 60)
    coverage_stats = predictor.verify_coverage(test_scores, test_labels)

    logger.info("Coverage verification:")
    for k, v in coverage_stats.items():
        logger.info("  %-30s %.4f", k, v)

    # Warn if coverage is below guarantee
    empirical  = coverage_stats["empirical_coverage"]
    target     = coverage_stats["target_coverage"]
    coverage_ok = empirical >= target
    if not coverage_ok:
        logger.warning(
            "COVERAGE BELOW GUARANTEE: %.4f < %.4f  "
            "(theory guarantees this only happens with probability alpha=%.2f)",
            empirical, target, config.conformal.alpha,
        )
    else:
        logger.info(
            "Coverage OK: %.4f >= %.4f  ✓", empirical, target,
        )

    # ── 8. produce per-sample intervals for full test set ─────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Step 7/7 — Generating per-sample conformal intervals (test set)")
    logger.info("=" * 60)

    test_lower, test_upper, test_widths = predictor.predict_batch(test_scores)

    # Identify which windows would trigger escalation
    escalation_mask = predictor.should_escalate(test_widths)  # may be scalar bool

    # Handle both scalar-bool and array-bool from should_escalate
    if isinstance(escalation_mask, bool):
        escalation_mask = np.full(len(test_widths), escalation_mask, dtype=bool)
    else:
        escalation_mask = np.array(
            [predictor.should_escalate(w) for w in test_widths], dtype=bool
        )

    n_escalated = int(escalation_mask.sum())
    pct_escalated = 100.0 * n_escalated / len(test_widths)

    logger.info("Conformal interval summary (test set):")
    logger.info("  q_alpha (half-width)       : %.4f", predictor.q_alpha)
    logger.info("  Mean interval width        : %.4f", float(test_widths.mean()))
    logger.info("  Median interval width      : %.4f", float(np.median(test_widths)))
    logger.info("  Min / Max width            : %.4f / %.4f",
                float(test_widths.min()), float(test_widths.max()))
    logger.info(
        "  Windows requiring escalation : %d / %d  (%.1f%%)",
        n_escalated, len(test_widths), pct_escalated,
    )

    # Breakdown by positive/negative
    pos_mask = test_labels == 1
    neg_mask = ~pos_mask
    if pos_mask.any():
        logger.info("  Positive (sepsis) windows:")
        logger.info("    Mean score   : %.4f", float(test_scores[pos_mask].mean()))
        logger.info("    Mean width   : %.4f", float(test_widths[pos_mask].mean()))
        logger.info("    Escalations  : %d / %d",
                    int(escalation_mask[pos_mask].sum()), int(pos_mask.sum()))
    if neg_mask.any():
        logger.info("  Negative (non-sepsis) windows:")
        logger.info("    Mean score   : %.4f", float(test_scores[neg_mask].mean()))
        logger.info("    Mean width   : %.4f", float(test_widths[neg_mask].mean()))
        logger.info("    Escalations  : %d / %d",
                    int(escalation_mask[neg_mask].sum()), int(neg_mask.sum()))

    # ── 9. save outputs ───────────────────────────────────────────────────────

    # 9a. Per-sample test intervals
    intervals_path = output_dir / "conformal_test_intervals.npz"
    np.savez(
        intervals_path,
        risk_scores   = test_scores,
        lower         = test_lower,
        upper         = test_upper,
        widths        = test_widths,
        labels        = test_labels,
        escalation    = escalation_mask.astype(np.uint8),
    )
    logger.info("\nSaved test intervals → %s", intervals_path)

    # 9b. Val scores (useful for re-calibration experiments)
    val_scores_path = output_dir / "conformal_val_scores.npz"
    val_lower, val_upper, val_widths = predictor.predict_batch(val_scores)
    np.savez(
        val_scores_path,
        risk_scores = val_scores,
        lower       = val_lower,
        upper       = val_upper,
        widths      = val_widths,
        labels      = val_labels,
    )
    logger.info("Saved val scores     → %s", val_scores_path)

    # 9c. Calibration state (JSON) — includes q_alpha for orchestrator
    calibration_state = {
        "method"              : "split_conformal",
        "alpha"               : config.conformal.alpha,
        "coverage_guarantee"  : 1.0 - config.conformal.alpha,
        "q_alpha"             : float(predictor.q_alpha),
        "escalation_threshold": config.conformal.escalation_width_threshold,
        "checkpoint_used"     : str(checkpoint_path),
        "data_used"           : str(data_path),
        "calibration_stats"   : {k: (float(v) if isinstance(v, float) else int(v))
                                  for k, v in cal_stats.items()},
        "coverage_verification": {k: float(v) for k, v in coverage_stats.items()},
        "test_interval_summary": {
            "n_test_windows"         : int(len(test_scores)),
            "n_positive_windows"     : int(pos_mask.sum()),
            "n_negative_windows"     : int(neg_mask.sum()),
            "mean_width"             : float(test_widths.mean()),
            "median_width"           : float(np.median(test_widths)),
            "min_width"              : float(test_widths.min()),
            "max_width"              : float(test_widths.max()),
            "n_escalated"            : n_escalated,
            "pct_escalated"          : float(pct_escalated),
            "mean_score_positive"    : float(test_scores[pos_mask].mean()) if pos_mask.any() else None,
            "mean_score_negative"    : float(test_scores[neg_mask].mean()) if neg_mask.any() else None,
        },
        "coverage_ok"         : bool(coverage_ok),
        "files": {
            "intervals_npz"   : str(intervals_path),
            "val_scores_npz"  : str(val_scores_path),
        },
    }

    json_path = output_dir / "conformal_calibration.json"
    with open(json_path, "w") as f:
        json.dump(calibration_state, f, indent=2)
    logger.info("Saved calibration state → %s", json_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("CONFORMAL CALIBRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  q_alpha (±half-width)       : %.4f", predictor.q_alpha)
    logger.info("  Target coverage             : %.0f%%", target * 100)
    logger.info("  Empirical coverage (test)   : %.2f%%", empirical * 100)
    logger.info("  Mean interval width (test)  : %.4f", float(test_widths.mean()))
    logger.info("  Windows escalated (test)    : %.1f%%", pct_escalated)
    logger.info("  Coverage guarantee met?     : %s", "YES ✓" if coverage_ok else "NO ✗")
    logger.info("")
    logger.info("Output files:")
    logger.info("  %s", json_path)
    logger.info("  %s", intervals_path)
    logger.info("  %s", val_scores_path)
    logger.info("=" * 60)

    return calibration_state


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic smoke test (runs without GPU or MIMIC-IV data)
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test(output_dir: str = "data/processed") -> None:
    """Quick local test of the full calibration pipeline using synthetic data.

    Creates a temporary HDF5 file and a random LSTM checkpoint so the entire
    script can be verified without needing the real MIMIC-IV data or the GPU
    server.

    Usage:
        python3 scripts/run_conformal_calibration.py --synthetic
    """
    import h5py
    import tempfile

    logger.info("=" * 60)
    logger.info("SYNTHETIC SMOKE TEST")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)
    config = get_default_config()

    # ── synthetic windows ──────────────────────────────────────────────────────
    N_train, N_val, N_test = 2000, 400, 500
    # Make positive class have slightly higher feature values so the model
    # can produce somewhat meaningful scores even without training.
    def make_split(n, pos_rate=0.13):
        labels = (rng.random(n) < pos_rate).astype(np.float32)
        X = rng.standard_normal((n, 6, 12)).astype(np.float32)
        X[labels == 1] += 0.3   # slight signal
        return X, labels.astype(np.int8)

    X_train, y_train = make_split(N_train)
    X_val,   y_val   = make_split(N_val)
    X_test,  y_test  = make_split(N_test)

    tmp_h5 = Path(output_dir) / "_synthetic_conformal_test.h5"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_val",   data=X_val)
        f.create_dataset("y_val",   data=y_val)
        f.create_dataset("X_test",  data=X_test)
        f.create_dataset("y_test",  data=y_test)
    logger.info("Synthetic HDF5 written → %s", tmp_h5)

    # ── synthetic checkpoint ───────────────────────────────────────────────────
    model = SepsisLSTM(config.lstm)   # random weights — output will be ~0.5
    tmp_ckpt = Path(output_dir) / "_synthetic_lstm_checkpoint.pt"
    torch.save(
        {
            "epoch"            : 0,
            "model_state_dict" : model.state_dict(),
            "best_val_auroc"   : 0.5,
            "metrics"          : {},
        },
        tmp_ckpt,
    )
    logger.info("Synthetic checkpoint written → %s", tmp_ckpt)

    # ── run pipeline ──────────────────────────────────────────────────────────
    result = run_conformal_calibration(
        checkpoint_path=str(tmp_ckpt),
        data_path=str(tmp_h5),
        output_dir=output_dir,
        batch_size=64,
        device_str="cpu",
    )

    # ── cleanup temp files ────────────────────────────────────────────────────
    tmp_h5.unlink(missing_ok=True)
    tmp_ckpt.unlink(missing_ok=True)
    logger.info("Temp files cleaned up.")

    logger.info("\n✓ Synthetic smoke test passed!")
    logger.info(
        "  q_alpha=%.4f | coverage=%.4f | mean_width=%.4f",
        result["q_alpha"],
        result["coverage_verification"]["empirical_coverage"],
        result["test_interval_summary"]["mean_width"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run conformal calibration on trained LSTM risk scores"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lstm_best.pt",
        help="Path to LSTM checkpoint (default: checkpoints/lstm_best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/features.h5",
        help="Path to windowed HDF5 features (default: data/processed/features.h5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save calibration outputs (default: data/processed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (default: 512). Reduce if OOM on CPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' | 'cuda' | 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic smoke test (no real data needed)",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_test(output_dir=args.output_dir)
    else:
        run_conformal_calibration(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device_str=args.device,
        )
