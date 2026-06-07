"""
QuantumSepsis Shield — QCCP Runner (FIXED VERSION)
Runs Quantum-Calibrated Conformal Prediction using quantum kernel centroids.

FIXES:
- Limits test set to 5000 balanced samples (prevents infinite quantum computation)
- Adds progress logging for quantum predictions
- Adds time estimates
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_default_config
from src.models.quantum_kernel import QuantumKernelSepsis
from src.models.conformal import QuantumCalibratedConformal, ConformalSepsisPredictor
from src.models.lstm import SepsisLSTM
from src.data.dataset import SepsisDataset

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = get_default_config()

    logger.info("Loading embeddings...")
    qk = QuantumKernelSepsis(config.quantum, random_state=config.training.seed)
    data = qk.load_embeddings("data/processed/lstm_embeddings.npz")

    # Training subset (500 samples)
    X_train_sub, y_train_sub, _ = qk.balanced_subsample(
        data["train_embeddings"], data["train_labels"], max_samples=500
    )

    X_train_pca = qk.fit_pca(X_train_sub)
    X_val_pca = qk.transform_pca(data["val_embeddings"])
    X_test_pca = qk.transform_pca(data["test_embeddings"])

    qk.setup_qiskit_kernel()

    logger.info("Training quantum SVM...")
    qk.fit(X_train_pca, y_train_sub)

    logger.info("Computing sepsis centroids...")
    centroids = qk.get_centroids(X_train_pca, y_train_sub, n_centroids=5)
    logger.info("Centroids shape: %s", centroids.shape)

    logger.info("Setting up QCCP...")
    qccp = QuantumCalibratedConformal(config.conformal)
    qccp.set_quantum_kernel(kernel_fn=qk.compute_kernel_matrix, centroids=centroids)

    logger.info("Calibrating QCCP on validation embeddings...")
    n_cal = min(500, len(X_val_pca))
    cal_stats = qccp.calibrate_quantum(X_val_pca[:n_cal], data["val_labels"][:n_cal])
    logger.info("QCCP calibration: %s", cal_stats)

    logger.info("Loading LSTM for standard conformal baseline...")
    lstm = SepsisLSTM(config.lstm)
    ckpt = torch.load("checkpoints/lstm_best.pt", map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        lstm.load_state_dict(ckpt["model_state_dict"])
    else:
        lstm.load_state_dict(ckpt)
    lstm.eval()

    val_ds = SepsisDataset.from_hdf5("data/processed/features.h5", split="val")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    val_scores, val_labels = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            out = lstm(bx)
            val_scores.append(out["risk_score"].cpu().numpy())
            val_labels.append(by.numpy())

    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)

    standard = ConformalSepsisPredictor(config.conformal)
    standard.calibrate(val_scores, val_labels)

    # ===== FIX: Limit test set to 5000 balanced samples =====
    logger.info("=" * 60)
    logger.info("LIMITING TEST SET TO 5000 BALANCED SAMPLES")
    logger.info("(Full test set quantum prediction is computationally intractable)")
    logger.info("=" * 60)
    
    # Create balanced test subset
    test_pos_idx = np.where(data["test_labels"] == 1)[0]
    test_neg_idx = np.where(data["test_labels"] == 0)[0]
    
    n_test_samples = 5000
    n_pos = min(n_test_samples // 2, len(test_pos_idx))
    n_neg = min(n_test_samples // 2, len(test_neg_idx))
    
    np.random.seed(config.training.seed)
    selected_pos = np.random.choice(test_pos_idx, n_pos, replace=False)
    selected_neg = np.random.choice(test_neg_idx, n_neg, replace=False)
    test_subset_idx = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(test_subset_idx)
    
    X_test_subset = X_test_pca[test_subset_idx]
    y_test_subset = data["test_labels"][test_subset_idx]
    
    logger.info(f"Test subset: {len(X_test_subset)} samples ({n_pos} pos, {n_neg} neg)")
    logger.info(f"Starting quantum kernel predictions (this will take ~30-60 min)...")
    
    # Time the quantum prediction
    start_time = time.time()
    test_scores_qk, _ = qk.predict_scores(X_test_subset, batch_size=256)
    elapsed = time.time() - start_time
    
    logger.info(f"Quantum predictions completed in {elapsed/60:.1f} minutes")
    logger.info(f"Average time per sample: {elapsed/len(X_test_subset):.3f} seconds")

    # Get LSTM scores for the same subset
    test_ds = SepsisDataset.from_hdf5("data/processed/features.h5", split="test")
    
    # Extract only the subset indices
    logger.info("Loading LSTM scores for test subset...")
    test_lstm_scores_full = []
    with torch.no_grad():
        for bx, _ in DataLoader(test_ds, batch_size=512, shuffle=False):
            out = lstm(bx)
            test_lstm_scores_full.append(out["risk_score"].cpu().numpy())
    
    test_lstm_scores_full = np.concatenate(test_lstm_scores_full)
    test_lstm_scores = test_lstm_scores_full[test_subset_idx]

    logger.info("Computing conformal intervals...")
    std_lower, std_upper, std_widths = standard.predict_batch(test_lstm_scores)
    qccp_lower, qccp_upper, qccp_widths = qccp.predict_batch(test_scores_qk)

    std_mean = float(std_widths.mean())
    qccp_mean = float(qccp_widths.mean())
    width_reduction_pct = float(((std_mean - qccp_mean) / std_mean) * 100.0) if std_mean > 0 else 0.0

    qccp_coverage = qccp.verify_coverage(test_scores_qk, y_test_subset)

    results = {
        "kernel_backend": qk._kernel_name,
        "n_centroids": int(len(centroids)),
        "n_test_samples": int(len(X_test_subset)),
        "test_subset_info": {
            "total_available": int(len(X_test_pca)),
            "used_for_evaluation": int(len(X_test_subset)),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "reason": "Full test set quantum prediction computationally intractable"
        },
        "standard_conformal": {
            "q_alpha": float(standard.q_alpha),
            "mean_width": std_mean,
            "median_width": float(np.median(std_widths)),
        },
        "qccp": {
            "q_alpha": float(qccp.q_alpha),
            "mean_width": qccp_mean,
            "median_width": float(np.median(qccp_widths)),
        },
        "width_reduction_pct": width_reduction_pct,
        "calibration_stats": cal_stats,
        "qccp_coverage": qccp_coverage,
        "computation_time_minutes": round(elapsed / 60, 2),
    }

    out_path = Path("data/processed/qccp_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("QCCP vs Standard Conformal Comparison")
    print("=" * 60)
    print(f"  Kernel backend:       {qk._kernel_name}")
    print(f"  Test samples:         {len(X_test_subset)} (balanced subset)")
    print(f"  Computation time:     {elapsed/60:.1f} minutes")
    print(f"  Standard q_alpha:     {standard.q_alpha:.4f}")
    print(f"  QCCP q_alpha:         {qccp.q_alpha:.4f}")
    print(f"  Standard mean width:  {std_mean:.4f}")
    print(f"  QCCP mean width:      {qccp_mean:.4f}")
    print(f"  Width reduction:      {width_reduction_pct:.1f}%")
    print(f"  QCCP coverage:        {qccp_coverage['empirical_coverage']:.4f}")
    print("=" * 60)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
