"""
QuantumSepsis Shield — LSTM Hyperparameter Tuning Runner
=========================================================

Runs multiple LSTM configurations to close the AUROC gap vs XGBoost (0.8038).

Experiments (in order of effort):
  EXP-1: hidden_dim 128 → 256  (more capacity)
  EXP-2: n_layers   2   → 3   (deeper BiLSTM)
  EXP-3: focal_gamma 2.0 → 3.0 (harder focus on uncertain examples)
  EXP-4: prediction_horizon 4h → 2h (easier task, more pos windows)
  EXP-5: hidden_dim=256 + gamma=3.0 + undersample 10:1  (combined)

Each experiment:
  - Overrides the default config with the experiment's values
  - Trains LSTM (or runs synthetic quick-check locally)
  - Saves checkpoint to checkpoints/tuning/<exp_name>/lstm_best.pt
  - Saves metrics to data/processed/tuning_results.json

Usage:
    # On GPU server — run all experiments:
    python3 scripts/run_lstm_tuning.py --all

    # Run a single experiment:
    python3 scripts/run_lstm_tuning.py --exp exp1_hidden256

    # Quick syntax/logic check (no GPU, no data):
    python3 scripts/run_lstm_tuning.py --synthetic

    # List all experiments:
    python3 scripts/run_lstm_tuning.py --list
"""

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_default_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment registry
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "exp1_hidden256": {
        "description": "Increase hidden_dim 128→256. More LSTM capacity.",
        "overrides": {
            "lstm.hidden_dim"    : 256,
            "lstm.attention_dim" : 128,
            "training.use_wandb" : False,
        },
        "screen_name": "qs_tune_exp1",
    },
    "exp2_layers3": {
        "description": "3-layer BiLSTM instead of 2. Deeper temporal modelling.",
        "overrides": {
            "lstm.n_layers"      : 3,
            "training.use_wandb" : False,
        },
        "screen_name": "qs_tune_exp2",
    },
    "exp3_gamma3": {
        "description": "Focal loss gamma 2.0→3.0. Harder down-weighting of easy negatives.",
        "overrides": {
            "training.focal_gamma": 3.0,
            "training.use_wandb"  : False,
        },
        "screen_name": "qs_tune_exp3",
    },
    "exp4_horizon2h": {
        "description": "Predict sepsis 2h ahead instead of 4h. Easier task, higher pos rate.",
        "overrides": {
            # Note: requires re-running windowing with horizon=2
            # Flag here for documentation; windowing must be re-run manually
            "data.prediction_horizon_hours": 2,
            "training.use_wandb"           : False,
        },
        "screen_name": "qs_tune_exp4",
        "note": "Requires re-running: python3 scripts/run_windowing_real.py --horizon 2",
    },
    "exp5_combined": {
        "description": "hidden_dim=256 + gamma=3.0 + n_layers=3. Best-of-all combined.",
        "overrides": {
            "lstm.hidden_dim"     : 256,
            "lstm.attention_dim"  : 128,
            "lstm.n_layers"       : 3,
            "training.focal_gamma": 3.0,
            "training.use_wandb"  : False,
        },
        "screen_name": "qs_tune_exp5",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Config builder
# ══════════════════════════════════════════════════════════════════════════════

def apply_overrides(config, overrides: dict):
    """Apply dot-separated overrides to a Config object.

    Example: "lstm.hidden_dim" = 256  sets config.lstm.hidden_dim = 256
    """
    cfg = deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        obj   = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg


def build_experiment_config(exp_name: str):
    """Return a fully configured Config for a given experiment."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}. Run --list to see options.")
    base   = get_default_config()
    exp    = EXPERIMENTS[exp_name]
    config = apply_overrides(base, exp["overrides"])
    config.training.checkpoint_dir = f"checkpoints/tuning/{exp_name}"
    return config, exp


# ══════════════════════════════════════════════════════════════════════════════
# Run one experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    exp_name : str,
    data_path: str = "data/processed/features.h5",
    device   : str = "auto",
) -> dict:
    """Train LSTM with experiment config, return metrics dict."""
    import torch
    from src.data.dataset import create_dataloaders
    from src.training.train_lstm import LSTMTrainer

    config, exp = build_experiment_config(exp_name)

    logger.info("\n%s", "="*60)
    logger.info("EXPERIMENT: %s", exp_name)
    logger.info("  %s", exp["description"])
    logger.info("  Overrides: %s", exp["overrides"])
    if "note" in exp:
        logger.warning("  NOTE: %s", exp["note"])
    logger.info("="*60)

    # Create output dirs
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(data_path, config)

    # Train
    device_obj = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else
        (device if device != "auto" else "cpu")
    )
    trainer    = LSTMTrainer(config, device=str(device_obj))
    t0         = time.time()
    train_res  = trainer.train(train_loader, val_loader)
    test_res   = trainer.evaluate(test_loader)

    # Extract embeddings for quantum kernel
    logger.info("Extracting embeddings...")
    emb_dir = Path("data/processed") / "tuning" / exp_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    train_emb, train_labels = trainer.extract_embeddings(train_loader)
    val_emb,   val_labels   = trainer.extract_embeddings(val_loader)
    test_emb,  test_labels  = trainer.extract_embeddings(test_loader)
    np.savez(
        emb_dir / "lstm_embeddings.npz",
        train_embeddings=train_emb, train_labels=train_labels,
        val_embeddings  =val_emb,   val_labels  =val_labels,
        test_embeddings =test_emb,  test_labels =test_labels,
    )
    logger.info("Embeddings saved → %s", emb_dir / "lstm_embeddings.npz")

    result = {
        "exp_name"   : exp_name,
        "description": exp["description"],
        "overrides"  : exp["overrides"],
        "training_time_minutes": (time.time()-t0)/60,
        **train_res,
        **test_res,
    }

    logger.info("Experiment %s done | test_auroc=%.4f | best_val_auroc=%.4f",
                exp_name,
                result.get("test_auroc", float("nan")),
                result.get("best_val_auroc", float("nan")))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Run all experiments + save comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(
    data_path : str = "data/processed/features.h5",
    output_dir: str = "data/processed",
    device    : str = "auto",
) -> dict:
    """Run every experiment in sequence and save comparison table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp_name in EXPERIMENTS:
        try:
            result = run_experiment(exp_name, data_path, device)
            all_results[exp_name] = result
        except Exception as e:
            logger.error("Experiment %s failed: %s", exp_name, e)
            all_results[exp_name] = {"error": str(e)}

        # Save partial results after each experiment (crash-safe)
        json_path = output_dir / "tuning_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print comparison table
    logger.info("\n%s\nTUNING RESULTS COMPARISON\n%s", "="*60, "="*60)
    baseline_auroc = 0.7891   # original LSTM
    xgb_auroc      = 0.8038
    logger.info("  %-20s %-12s %-12s %-10s", "Experiment", "Test AUROC", "Val AUROC", "vs baseline")
    logger.info("  %-20s %-12s %-12s", "baseline (LSTM)", f"{baseline_auroc:.4f}", "0.7601")
    logger.info("  %-20s %-12s %-12s", "XGBoost (target)", f"{xgb_auroc:.4f}", "—")

    for exp_name, res in all_results.items():
        if "error" in res:
            logger.info("  %-20s FAILED: %s", exp_name, res["error"])
            continue
        test_a = res.get("test_auroc",     float("nan"))
        val_a  = res.get("best_val_auroc", float("nan"))
        diff   = test_a - baseline_auroc
        logger.info("  %-20s %-12.4f %-12.4f %+.4f",
                    exp_name, test_a, val_a, diff)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic test (validates config building + override logic)
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test(output_dir: str = "data/processed") -> None:
    """Validate experiment configs without training."""
    import h5py
    import torch
    import tempfile
    from src.models.lstm import SepsisLSTM

    logger.info("="*60)
    logger.info("SYNTHETIC VALIDATION — Config building + override logic")
    logger.info("="*60)

    errors = []
    for exp_name, exp in EXPERIMENTS.items():
        try:
            config, _ = build_experiment_config(exp_name)
            model     = SepsisLSTM(config.lstm)
            rng       = np.random.default_rng(0)
            x         = torch.FloatTensor(rng.standard_normal((4, 6, 12)).astype(np.float32))
            out       = model(x)
            assert "risk_score"  in out
            assert "embedding"   in out
            assert out["embedding"].shape[-1] == config.lstm.embedding_dim
            logger.info("  ✓ %-20s | hidden=%d layers=%d gamma=%.1f",
                        exp_name,
                        config.lstm.hidden_dim,
                        config.lstm.n_layers,
                        config.training.focal_gamma)
        except Exception as e:
            logger.error("  ✗ %-20s | %s", exp_name, e)
            errors.append(exp_name)

    if errors:
        logger.error("FAILED experiments: %s", errors)
        sys.exit(1)
    else:
        logger.info("\n✓ All %d experiment configs validated!", len(EXPERIMENTS))

    # Validate override logic
    base = get_default_config()
    cfg  = apply_overrides(base, {"lstm.hidden_dim": 999, "lstm.n_layers": 5})
    assert cfg.lstm.hidden_dim == 999
    assert cfg.lstm.n_layers   == 5
    assert base.lstm.hidden_dim != 999   # must not mutate base
    logger.info("✓ apply_overrides does not mutate base config")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Hyperparameter Tuning")
    parser.add_argument("--all",       action="store_true", help="Run all experiments")
    parser.add_argument("--exp",       type=str, help="Run one experiment by name")
    parser.add_argument("--list",      action="store_true", help="List all experiments")
    parser.add_argument("--synthetic", action="store_true", help="Config validation only")
    parser.add_argument("--data",      default="data/processed/features.h5")
    parser.add_argument("--output-dir",default="data/processed")
    parser.add_argument("--device",    default="auto")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:25s}  {exp['description']}")
            if "note" in exp:
                print(f"  {'':25s}  ⚠  {exp['note']}")
        sys.exit(0)

    if args.synthetic:
        run_synthetic_test(args.output_dir)
    elif args.exp:
        result = run_experiment(args.exp, args.data, args.device)
        json_path = Path(args.output_dir) / f"tuning_{args.exp}.json"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved → %s", json_path)
    elif args.all:
        run_all_experiments(args.data, args.output_dir, args.device)
    else:
        parser.print_help()
