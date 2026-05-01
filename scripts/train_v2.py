"""
QuantumSepsis Shield — V2 Training Pipeline
=============================================

End-to-end training script for the enhanced model:
  1. Load windowed HDF5 (N, 6, 12)
  2. Enrich features: (N, 6, 12) → (N, 6, 33) with derived features
  3. Normalize derived features (z-score from train set)
  4. Train SepsisLSTMv2 with AsymmetricFocalLoss
  5. Extract embeddings for quantum kernel V2
  6. Save checkpoint + embeddings + normalization stats

Usage:
  # On GPU server with real data:
  screen -S train_v2
  cd ~/QuantumSepsis && export PYTHONPATH=.
  CUDA_VISIBLE_DEVICES=0 python3 scripts/train_v2.py

  # Local smoke test:
  python3 scripts/train_v2.py --synthetic

  # Custom args:
  python3 scripts/train_v2.py --data data/processed/features.h5 \\
      --epochs 100 --batch-size 256 --lr 0.001
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feature_engineering_v2 import (
    enrich_batch,
    normalize_derived_features,
    V2_FEATURE_NAMES,
    N_FEATURES_V2,
)
from src.models.lstm_v2 import SepsisLSTMv2
from src.models.losses import AsymmetricFocalLoss
from src.evaluation.metrics import compute_all_metrics, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Enrichment
# ══════════════════════════════════════════════════════════════════════════════

def load_and_enrich(data_path: str) -> dict:
    """Load HDF5, enrich features 12→33, normalize derived features."""
    import h5py

    logger.info("Loading data from %s", data_path)
    with h5py.File(data_path, "r") as f:
        X_train = f["X_train"][:].astype(np.float32)
        y_train = f["y_train"][:].astype(np.int8)
        X_val = f["X_val"][:].astype(np.float32)
        y_val = f["y_val"][:].astype(np.int8)
        X_test = f["X_test"][:].astype(np.float32)
        y_test = f["y_test"][:].astype(np.int8)

    logger.info("Raw shapes: train=%s, val=%s, test=%s", X_train.shape, X_val.shape, X_test.shape)

    # Enrich: 12 → 33 features
    logger.info("Enriching features (12 → %d)...", N_FEATURES_V2)
    X_train_v2 = enrich_batch(X_train)
    X_val_v2 = enrich_batch(X_val)
    X_test_v2 = enrich_batch(X_test)

    # Normalize derived features
    logger.info("Normalizing derived features...")
    X_train_v2, X_val_v2, X_test_v2, norm_stats = normalize_derived_features(
        X_train_v2, X_val_v2, X_test_v2
    )

    return {
        "X_train": X_train_v2, "y_train": y_train,
        "X_val": X_val_v2, "y_val": y_val,
        "X_test": X_test_v2, "y_test": y_test,
        "norm_stats": norm_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_v2(
    data_path: str = "data/processed/features.h5",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device_str: str = "auto",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "data/processed",
) -> dict:
    """Full V2 training pipeline."""

    # ── Device ─────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # ── Data ───────────────────────────────────────────────────────────────
    data = load_and_enrich(data_path)

    train_ds = TensorDataset(
        torch.from_numpy(data["X_train"]),
        torch.from_numpy(data["y_train"].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data["X_val"]),
        torch.from_numpy(data["y_val"].astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(data["X_test"]),
        torch.from_numpy(data["y_test"].astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Train: %d windows | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    # ── Model ──────────────────────────────────────────────────────────────
    model = SepsisLSTMv2(
        input_size=N_FEATURES_V2,
        seq_len=6,
        conv_channels=32,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        dropout=0.3,
        embedding_dim=16,
    ).to(device)
    logger.info("\n%s", model.summary())

    # ── Loss / Optimizer / Scheduler ───────────────────────────────────────
    criterion = AsymmetricFocalLoss(alpha_pos=0.9, alpha_neg=0.1, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # — Train —
        model.train()
        train_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out["logits"], by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # — Validate —
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                out = model(bx)
                val_scores.append(out["risk_score"].cpu().numpy())
                val_labels.append(by.numpy())

        val_scores = np.concatenate(val_scores)
        val_labels = np.concatenate(val_labels)
        val_metrics = compute_all_metrics(val_labels, val_scores, prefix="val_")
        val_auroc = val_metrics["val_auroc"]

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d | loss=%.4f | val_auroc=%.4f | lr=%.2e | %.1fs",
            epoch, epochs, train_loss, val_auroc, lr_now, elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auroc,
            "lr": lr_now,
        })

        # — Early stopping —
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = Path(checkpoint_dir) / "lstm_v2_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auroc": best_val_auroc,
                "model_config": {
                    "input_size": N_FEATURES_V2,
                    "seq_len": 6,
                    "conv_channels": 32,
                    "hidden_dim": 128,
                    "n_layers": 2,
                    "n_heads": 4,
                    "dropout": 0.3,
                    "embedding_dim": 16,
                },
            }, ckpt_path)
            logger.info("  ★ New best! Saved to %s", ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # ── Load best model ────────────────────────────────────────────────────
    ckpt = torch.load(Path(checkpoint_dir) / "lstm_v2_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded best model from epoch %d (val_auroc=%.4f)", best_epoch, best_val_auroc)

    # ── Test evaluation ────────────────────────────────────────────────────
    test_scores, test_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            test_scores.append(out["risk_score"].cpu().numpy())
            test_labels.append(by.numpy())

    test_scores = np.concatenate(test_scores)
    test_labels = np.concatenate(test_labels)
    test_metrics = compute_all_metrics(test_labels, test_scores, prefix="test_")

    logger.info("\n%s", format_metrics(test_metrics, "V2 Model — Test Results"))

    # ── Extract embeddings ─────────────────────────────────────────────────
    logger.info("Extracting V2 embeddings for quantum kernel...")
    embeddings = {}
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        emb_list, lab_list = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                e = model.extract_embeddings(bx)
                emb_list.append(e.cpu().numpy())
                lab_list.append(by.numpy())
        embeddings[f"{name}_embeddings"] = np.concatenate(emb_list).astype(np.float32)
        embeddings[f"{name}_labels"] = np.concatenate(lab_list).astype(np.int8)

    emb_path = Path(output_dir) / "lstm_v2_embeddings.npz"
    np.savez(emb_path, **embeddings)
    logger.info("V2 embeddings saved to %s", emb_path)

    # ── Save normalization stats ───────────────────────────────────────────
    v2_stats_path = Path(output_dir) / "v2_normalization_stats.json"
    stats_json = {
        "derived_mean": data["norm_stats"]["derived_mean"].tolist(),
        "derived_std": data["norm_stats"]["derived_std"].tolist(),
        "derived_feature_names": list(data["norm_stats"]["derived_feature_names"]),
        "all_feature_names": V2_FEATURE_NAMES,
        "n_raw_features": 12,
        "n_derived_features": N_FEATURES_V2 - 12,
        "n_total_features": N_FEATURES_V2,
    }
    with open(v2_stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    logger.info("V2 normalization stats saved to %s", v2_stats_path)

    # ── Save training history ──────────────────────────────────────────────
    results = {
        "model": "SepsisLSTMv2",
        "n_features": N_FEATURES_V2,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "training_history": history,
        **test_metrics,
    }

    results_path = Path(output_dir) / "v2_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Training results saved to %s", results_path)

    logger.info("\n" + "=" * 60)
    logger.info("V2 TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best epoch:     %d", best_epoch)
    logger.info("  Val AUROC:      %.4f", best_val_auroc)
    logger.info("  Test AUROC:     %.4f", test_metrics["test_auroc"])
    logger.info("  Test AUPRC:     %.4f", test_metrics["test_auprc"])
    logger.info("  Checkpoint:     %s", Path(checkpoint_dir) / "lstm_v2_best.pt")
    logger.info("  Embeddings:     %s", emb_path)
    logger.info("=" * 60)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic smoke test
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic():
    """Quick local test without real data or GPU."""
    import h5py, tempfile

    logger.info("=" * 60)
    logger.info("SYNTHETIC SMOKE TEST — V2 Training")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)

    def make_split(n, pos_rate=0.13):
        labels = (rng.random(n) < pos_rate).astype(np.float32)
        X = rng.standard_normal((n, 6, 12)).astype(np.float32)
        X[labels == 1] += 0.3
        return X, labels.astype(np.int8)

    X_train, y_train = make_split(2000)
    X_val, y_val = make_split(400)
    X_test, y_test = make_split(500)

    tmp_h5 = Path("data/processed/_synthetic_v2_features.h5")
    tmp_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_val", data=X_val)
        f.create_dataset("y_val", data=y_val)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)

    results = train_v2(
        data_path=str(tmp_h5),
        epochs=5,
        batch_size=64,
        lr=1e-3,
        patience=3,
        device_str="cpu",
    )

    tmp_h5.unlink(missing_ok=True)
    logger.info("\n✓ Synthetic V2 smoke test passed! Test AUROC: %.4f", results["test_auroc"])


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SepsisLSTMv2 (enhanced model)")
    parser.add_argument("--data", type=str, default="data/processed/features.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
    else:
        train_v2(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device_str=args.device,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )
