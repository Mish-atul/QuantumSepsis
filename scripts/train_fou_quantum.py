"""
Train FOU Quantum Kernel
=========================
Training script for FOU quantum kernel using One-vs-Rest QSVM.

Usage:
    python scripts/train_fou_quantum.py --embeddings data/processed/fou/fou_lstm_embeddings.npz --max-samples 2000
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.quantum_kernel_fou import FouQuantumKernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train FOU Quantum Kernel")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to LSTM embeddings')
    parser.add_argument('--max-samples', type=int, default=2000, help='Max samples for training (500 per class)')
    parser.add_argument('--n-qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--reps', type=int, default=2, help='Feature map repetitions')
    parser.add_argument('--output-dir', type=str, default='data/processed/fou', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings}...")
    data = np.load(args.embeddings)

    X_train = data['train_embeddings']
    y_train = data['train_labels']
    X_test = data['test_embeddings']
    y_test = data['test_labels']

    logger.info(f"Train embeddings: {X_train.shape}")
    logger.info(f"Test embeddings: {X_test.shape}")
    logger.info(f"Train label distribution: {np.bincount(y_train)}")

    # Initialize quantum kernel
    logger.info(f"Initializing quantum kernel ({args.n_qubits} qubits, {args.reps} reps)...")
    qkernel = FouQuantumKernel(
        n_qubits=args.n_qubits,
        feature_map="ZZFeatureMap",
        entanglement="linear",
        reps=args.reps,
        backend="aer_simulator"
    )

    # Train quantum kernel
    max_samples_per_class = args.max_samples // 4  # 4 classes
    logger.info(f"Training quantum kernel (max {max_samples_per_class} samples per class)...")
    qkernel.fit(X_train, y_train, max_samples_per_class=max_samples_per_class)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    metrics = qkernel.evaluate(X_test, y_test)

    # Print results
    logger.info("\n=== FOU Quantum Kernel Results ===")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"\nPer-class F1:")
    logger.info(f"  Class 0 (No FOU): {metrics['class_0_f1']:.4f}")
    logger.info(f"  Class 1 (Infectious): {metrics['class_1_f1']:.4f}")
    logger.info(f"  Class 2 (Non-infectious): {metrics['class_2_f1']:.4f}")
    logger.info(f"  Class 3 (Undiagnosed): {metrics['class_3_f1']:.4f}")

    # Save results
    results = {
        "n_qubits": args.n_qubits,
        "reps": args.reps,
        "max_samples": args.max_samples,
        "train_samples": len(qkernel.X_train_reduced),
        "test_samples": len(X_test),
        "metrics": metrics
    }

    results_path = output_dir / "fou_quantum_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Get test predictions and probabilities
    logger.info("\nGenerating test predictions...")
    y_pred = qkernel.predict(X_test)
    y_proba = qkernel.predict_proba(X_test)

    # Save predictions
    predictions_path = output_dir / "fou_quantum_predictions.npz"
    np.savez(
        predictions_path,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )
    logger.info(f"Predictions saved to {predictions_path}")

    logger.info("\n✓ FOU quantum kernel training complete")


if __name__ == "__main__":
    main()
