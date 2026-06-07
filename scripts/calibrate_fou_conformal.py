"""
Calibrate FOU Conformal Prediction
===================================
Calibrate multi-class conformal prediction for FOU detection.

Usage:
    python scripts/calibrate_fou_conformal.py --model checkpoints/fou/fou_lstm_best.pt --data data/processed/fou/fou_features.h5
"""

import sys
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm_fou import FouLSTM
from src.models.conformal_fou import MultiClassConformalPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Calibrate FOU Conformal Prediction")
    parser.add_argument('--model', type=str, required=True, help='Path to trained FOU LSTM model')
    parser.add_argument('--data', type=str, required=True, help='Path to FOU features HDF5')
    parser.add_argument('--alpha', type=float, default=0.10, help='Miscoverage rate (1-α coverage)')
    parser.add_argument('--output-dir', type=str, default='data/processed/fou', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = FouLSTM(
        input_size=config.lstm.input_size,
        seq_len=config.lstm.seq_len,
        hidden_dim=config.lstm.hidden_dim,
        n_layers=config.lstm.n_layers,
        bidirectional=config.lstm.bidirectional,
        dropout=config.lstm.dropout,
        attention_dim=config.lstm.attention_dim,
        fc1_dim=config.lstm.fc1_dim,
        embedding_dim=config.lstm.embedding_dim,
        n_classes=config.lstm.n_classes
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("Model loaded successfully")

    # Load data
    logger.info(f"Loading data from {args.data}...")
    with h5py.File(args.data, 'r') as f:
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]

    logger.info(f"Val: {X_val.shape}, Test: {X_test.shape}")

    # Create dataloaders
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Get predictions on validation set (for calibration)
    logger.info("Getting validation predictions for calibration...")
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            _, probs, _, _ = model(X_batch)
            val_probs.append(probs.cpu().numpy())
            val_labels.append(y_batch.numpy())

    val_probs = np.vstack(val_probs)
    val_labels = np.hstack(val_labels)

    logger.info(f"Validation predictions: {val_probs.shape}")

    # Calibrate conformal predictor
    logger.info(f"\nCalibrating conformal predictor (α={args.alpha})...")
    conformal = MultiClassConformalPredictor(alpha=args.alpha, n_classes=4)
    q_alpha = conformal.calibrate(val_probs, val_labels)

    logger.info(f"Calibrated q_alpha: {q_alpha:.4f}")

    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_metrics = conformal.evaluate(val_probs, val_labels)

    logger.info(f"Validation metrics:")
    logger.info(f"  Coverage: {val_metrics['coverage']:.4f} (target: {1-args.alpha:.4f})")
    logger.info(f"  Avg set size: {val_metrics['avg_set_size']:.2f}")
    logger.info(f"  Singleton rate: {val_metrics['singleton_rate']:.4f}")
    logger.info(f"  Empty rate: {val_metrics['empty_rate']:.4f}")
    logger.info(f"  Full rate: {val_metrics['full_rate']:.4f}")

    # Get predictions on test set
    logger.info("\nGetting test predictions...")
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            _, probs, _, _ = model(X_batch)
            test_probs.append(probs.cpu().numpy())
            test_labels.append(y_batch.numpy())

    test_probs = np.vstack(test_probs)
    test_labels = np.hstack(test_labels)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = conformal.evaluate(test_probs, test_labels)

    logger.info(f"Test metrics:")
    logger.info(f"  Coverage: {test_metrics['coverage']:.4f} (target: {1-args.alpha:.4f})")
    logger.info(f"  Avg set size: {test_metrics['avg_set_size']:.2f}")
    logger.info(f"  Singleton rate: {test_metrics['singleton_rate']:.4f}")
    logger.info(f"  Empty rate: {test_metrics['empty_rate']:.4f}")
    logger.info(f"  Full rate: {test_metrics['full_rate']:.4f}")

    # Get prediction sets for test set
    prediction_sets, set_sizes = conformal.predict(test_probs)

    # Save calibration results
    results = {
        "alpha": args.alpha,
        "q_alpha": float(q_alpha),
        "target_coverage": 1 - args.alpha,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()}
    }

    results_path = output_dir / "fou_conformal_calibration.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nCalibration results saved to {results_path}")

    # Save prediction sets
    predictions_path = output_dir / "fou_conformal_predictions.npz"
    np.savez(
        predictions_path,
        test_probs=test_probs,
        test_labels=test_labels,
        prediction_sets=np.array([np.array(ps) for ps in prediction_sets], dtype=object),
        set_sizes=set_sizes
    )
    logger.info(f"Prediction sets saved to {predictions_path}")

    logger.info("\n✓ FOU conformal calibration complete")


if __name__ == "__main__":
    main()
