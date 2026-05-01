"""
QuantumSepsis Shield — QCCP Runner
Runs Quantum-Calibrated Conformal Prediction using quantum kernel centroids.
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path

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
    ckpt = torch.load("checkpoints/lstm_best.pt", map_location="cpu")
    lstm.load_state_dict(ckpt["model_state_dict"])
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

    test_scores_qk, _ = qk.predict_scores(X_test_pca, batch_size=256)

    test_ds = SepsisDataset.from_hdf5("data/processed/features.h5", split="test")
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    test_lstm_scores = []

    with torch.no_grad():
        for bx, _ in test_loader:
            out = lstm(bx)
            test_lstm_scores.append(out["risk_score"].cpu().numpy())

    test_lstm_scores = np.concatenate(test_lstm_scores)

    std_lower, std_upper, std_widths = standard.predict_batch(test_lstm_scores)
    qccp_lower, qccp_upper, qccp_widths = qccp.predict_batch(test_scores_qk)

    std_mean = float(std_widths.mean())
    qccp_mean = float(qccp_widths.mean())
    width_reduction_pct = float(((std_mean - qccp_mean) / std_mean) * 100.0) if std_mean > 0 else 0.0

    qccp_coverage = qccp.verify_coverage(test_scores_qk, data["test_labels"])

    results = {
        "kernel_backend": qk._kernel_name,
        "n_centroids": int(len(centroids)),
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
    }

    out_path = Path("data/processed/qccp_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("QCCP vs Standard Conformal Comparison")
    print("=" * 60)
    print(f"  Kernel backend:       {qk._kernel_name}")
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
