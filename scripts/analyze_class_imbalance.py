"""
QuantumSepsis Shield — Class Imbalance & AUPRC Investigation
=============================================================

Investigates why AUPRC is ~0.05 across all models and proposes fixes.

Analysis:
  1. Actual positive rate in windowed HDF5 (train/val/test)
  2. Class distribution per split
  3. AUPRC vs AUROC tradeoff explanation
  4. Suggested resampling ratios
  5. Focal loss gamma sensitivity analysis

Usage:
    python3 scripts/analyze_class_imbalance.py
    python3 scripts/analyze_class_imbalance.py --synthetic
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load label arrays from HDF5
# ══════════════════════════════════════════════════════════════════════════════

def load_label_stats(data_path: str) -> dict:
    """Load y_train / y_val / y_test from features.h5 and compute class stats."""
    import h5py

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HDF5 not found: {path}\n"
            "Run windowing first:\n  python3 scripts/run_windowing_real.py"
        )

    stats = {}
    with h5py.File(path, "r") as f:
        for split in ["train", "val", "test"]:
            y_key = f"y_{split}"
            if y_key not in f:
                logger.warning("Key %s missing from HDF5, skipping.", y_key)
                continue
            y = f[y_key][:]
            n         = len(y)
            n_pos     = int(y.sum())
            n_neg     = n - n_pos
            pos_rate  = n_pos / max(n, 1)
            # Random baseline AUPRC = positive rate
            auprc_random = pos_rate

            stats[split] = {
                "n_total"       : n,
                "n_positive"    : n_pos,
                "n_negative"    : n_neg,
                "positive_rate" : round(pos_rate, 6),
                "neg_pos_ratio" : round(n_neg / max(n_pos, 1), 1),
                "auprc_random_baseline": round(auprc_random, 6),
            }

            logger.info(
                "[%s]  n=%d  pos=%d (%.4f%%)  neg=%d  ratio=%.1f:1  "
                "AUPRC_random=%.4f",
                split.upper(), n, n_pos, pos_rate*100, n_neg,
                n_neg / max(n_pos, 1), auprc_random,
            )

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute recommendations
# ══════════════════════════════════════════════════════════════════════════════

def compute_recommendations(label_stats: dict) -> dict:
    """Generate actionable recommendations to improve AUPRC."""

    recs = {}
    for split, stats in label_stats.items():
        pos_rate  = stats["positive_rate"]
        ratio     = stats["neg_pos_ratio"]

        # Undersampling recommendation: keep all positives, subsample negatives
        # Target ratio for training: ~5:1 or ~10:1 (much better than 80:1)
        target_ratio  = 10.0
        n_pos         = stats["n_positive"]
        n_neg_keep    = int(n_pos * target_ratio)
        n_after_us    = n_pos + n_neg_keep
        pct_kept      = 100 * n_after_us / max(stats["n_total"], 1)

        # Focal loss gamma recommendation
        # Higher class imbalance → higher gamma needed to down-weight easy negatives
        gamma_rec = 2.0 if ratio < 20 else (3.0 if ratio < 50 else 4.0)

        # scale_pos_weight for XGBoost / class_weight for SVM
        scale_pos_weight = round(ratio, 1)

        recs[split] = {
            "current_pos_rate"       : pos_rate,
            "current_neg_pos_ratio"  : ratio,
            "auprc_random_baseline"  : stats["auprc_random_baseline"],
            "undersampling": {
                "target_ratio"       : target_ratio,
                "n_neg_to_keep"      : n_neg_keep,
                "total_after_us"     : n_after_us,
                "pct_data_kept"      : round(pct_kept, 1),
                "note"               : f"Keep all {n_pos} positives, sample {n_neg_keep} negatives"
            },
            "focal_loss": {
                "recommended_gamma"  : gamma_rec,
                "current_gamma"      : 2.0,
                "note"               : f"Increase gamma to {gamma_rec} to down-weight easy negatives"
            },
            "xgboost": {
                "scale_pos_weight"   : scale_pos_weight,
                "note"               : f"scale_pos_weight={scale_pos_weight} already handles this"
            },
            "prediction_horizon": {
                "current_hours"      : 4,
                "try_hours"          : [2, 3],
                "note"               : "Shorter horizon → more windows near onset → higher pos rate"
            },
        }

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Gamma sensitivity simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_focal_gamma_sensitivity(pos_rate: float) -> list:
    """Simulate how different gamma values weight easy vs hard examples.

    Returns list of dicts with gamma → effective weight ratios.
    """
    gammas  = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    results = []

    # Easy negative: p(wrong) = 0.05   Hard negative: p(wrong) = 0.4
    # Easy positive: p(wrong) = 0.05   Hard positive: p(wrong) = 0.4

    for gamma in gammas:
        easy_neg_weight = (0.05 ** gamma)   # well-classified negative
        hard_neg_weight = (0.40 ** gamma)   # hard-to-classify negative
        easy_pos_weight = (0.05 ** gamma)
        hard_pos_weight = (0.40 ** gamma)

        results.append({
            "gamma"           : gamma,
            "easy_neg_weight" : round(easy_neg_weight, 6),
            "hard_neg_weight" : round(hard_neg_weight, 6),
            "easy_pos_weight" : round(easy_pos_weight, 6),
            "hard_pos_weight" : round(hard_pos_weight, 6),
            "hard_vs_easy_ratio": round(hard_neg_weight / max(easy_neg_weight, 1e-9), 2),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_class_imbalance(
    data_path : str = "data/processed/features.h5",
    output_dir: str = "data/processed",
) -> dict:
    """Full class imbalance analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Label stats
    logger.info("\n%s\nStep 1/3 — Loading label distributions\n%s", "="*60, "="*60)
    label_stats = load_label_stats(data_path)

    # Step 2 — Recommendations
    logger.info("\n%s\nStep 2/3 — Computing recommendations\n%s", "="*60, "="*60)
    recommendations = compute_recommendations(label_stats)

    # Step 3 — Focal gamma sensitivity
    logger.info("\n%s\nStep 3/3 — Focal gamma sensitivity analysis\n%s", "="*60, "="*60)
    train_pos_rate = label_stats.get("train", {}).get("positive_rate", 0.05)
    gamma_analysis = simulate_focal_gamma_sensitivity(train_pos_rate)

    logger.info("Focal Loss gamma sensitivity (easy/hard weight ratios):")
    logger.info("  %-8s %-14s %-14s %-14s", "gamma", "easy_neg", "hard_neg", "hard/easy")
    for row in gamma_analysis:
        logger.info("  %-8.1f %-14.6f %-14.6f %-14.2f",
                    row["gamma"], row["easy_neg_weight"],
                    row["hard_neg_weight"], row["hard_vs_easy_ratio"])

    # Log recommendations for train split
    if "train" in recommendations:
        rec = recommendations["train"]
        logger.info("\n--- Recommendations (Train split) ---")
        logger.info("Current positive rate : %.4f%%", rec["current_pos_rate"]*100)
        logger.info("Current neg:pos ratio : %.1f:1", rec["current_neg_pos_ratio"])
        logger.info("AUPRC random baseline : %.4f", rec["auprc_random_baseline"])
        us = rec["undersampling"]
        logger.info("Undersampling: keep %d neg + %d pos = %d total (%.1f%% of data)",
                    us["n_neg_to_keep"], rec.get("n_pos",0),
                    us["total_after_us"], us["pct_data_kept"])
        logger.info("Focal gamma rec       : %.1f", rec["focal_loss"]["recommended_gamma"])
        logger.info("Try prediction horizon: %s hours (currently 4h)",
                    rec["prediction_horizon"]["try_hours"])

    result = {
        "label_stats"     : label_stats,
        "recommendations" : recommendations,
        "gamma_analysis"  : gamma_analysis,
        "summary": {
            "root_cause": (
                "AUPRC is low because sliding-window labelling creates severe "
                "class imbalance. A sepsis patient contributes ~4-6 positive windows "
                "out of potentially 100s of ICU hours. Most windows are negative."
            ),
            "fixes_ranked_by_effort": [
                "1. [Easy] Undersample negative windows at 10:1 ratio in DataLoader",
                "2. [Easy] Increase focal loss gamma from 2.0 → 3.0 or 4.0",
                "3. [Medium] Reduce prediction_horizon_hours from 4 → 2",
                "4. [Medium] Report stay-level metrics (any window CRITICAL → positive)",
                "5. [Hard] SMOTE on training windows in embedding space",
            ],
        }
    }

    json_path = output_dir / "class_imbalance_analysis.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\n%s\nANALYSIS COMPLETE\n%s", "="*60, "="*60)
    logger.info("Saved → %s", json_path)
    logger.info("\nRoot cause: %s", result["summary"]["root_cause"])
    logger.info("\nFixes (ranked by effort):")
    for fix in result["summary"]["fixes_ranked_by_effort"]:
        logger.info("  %s", fix)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test(output_dir: str = "data/processed") -> None:
    import h5py
    logger.info("="*60)
    logger.info("SYNTHETIC SMOKE TEST — Class Imbalance Analysis")
    logger.info("="*60)

    rng = np.random.default_rng(0)

    def make_split(n, pos_rate=0.05):
        y = (rng.random(n) < pos_rate).astype(np.int8)
        X = rng.standard_normal((n, 6, 12)).astype(np.float32)
        return X, y

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir   = Path(tmpdir)
        h5_path  = tmpdir / "features.h5"

        with h5py.File(h5_path, "w") as f:
            for split, n, pr in [("train",4000,0.04),("val",800,0.05),("test",1000,0.05)]:
                X, y = make_split(n, pr)
                f.create_dataset(f"X_{split}", data=X)
                f.create_dataset(f"y_{split}", data=y)

        result = analyze_class_imbalance(
            data_path  = str(h5_path),
            output_dir = str(tmpdir / "out"),
        )

        assert "label_stats" in result
        assert "recommendations" in result
        assert "gamma_analysis" in result
        assert len(result["gamma_analysis"]) == 6   # 6 gamma values
        assert (tmpdir / "out" / "class_imbalance_analysis.json").exists()

        logger.info("\n✓ Synthetic smoke test passed!")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class Imbalance & AUPRC Analysis")
    parser.add_argument("--data",       default="data/processed/features.h5")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--synthetic",  action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_test(output_dir=args.output_dir)
    else:
        analyze_class_imbalance(data_path=args.data, output_dir=args.output_dir)
