"""
Quantum Kernel Training for QuantumSepsis
==========================================

Trains QSVM on LSTM embeddings using quantum kernel methods.
Uses smaller subset for faster training.

Author: QuantumSepsis Team
Date: May 12, 2026
"""

import sys
import logging
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.quantum_kernel import QuantumKernelSepsis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train quantum kernel")
    parser.add_argument('--embeddings', type=str,
                        default='data/processed/sepsis/hierarchical_lstm_embeddings.npz')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Training samples (balanced)')
    parser.add_argument('--output-dir', type=str, default='data/processed/sepsis')
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['rbf', 'quantum'],
                        help='Kernel type (quantum requires Qiskit)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("QUANTUM KERNEL TRAINING")
    logger.info("="*70)
    logger.info(f"Embeddings: {args.embeddings}")
    logger.info(f"Training samples: {args.samples}")
    logger.info(f"Kernel: {args.kernel}")
    
    # Load embeddings
    logger.info("\nLoading embeddings...")
    data = np.load(args.embeddings)
    
    X_train = data['train_embeddings']
    y_train = data['train_labels']
    X_val = data['val_embeddings']
    y_val = data['val_labels']
    X_test = data['test_embeddings']
    y_test = data['test_labels']
    
    logger.info(f"Train: {X_train.shape}, positives: {y_train.sum()}")
    logger.info(f"Val:   {X_val.shape}, positives: {y_val.sum()}")
    logger.info(f"Test:  {X_test.shape}, positives: {y_test.sum()}")
    
    # Initialize quantum kernel model
    logger.info("\nInitializing quantum kernel model...")
    from src.config import Config
    config = Config()
    
    qk_model = QuantumKernelSepsis(
        config=config.quantum,
    )
    
    # Balanced subsample for training
    logger.info(f"\nCreating balanced subsample ({args.samples} samples)...")
    X_train_sub, y_train_sub, _ = qk_model.balanced_subsample(
        X_train, y_train, max_samples=args.samples
    )
    logger.info(f"Subsampled: {X_train_sub.shape}, positives: {y_train_sub.sum()}")
    
    # Fit PCA
    logger.info("\nFitting PCA (16-dim → 8-dim)...")
    qk_model.fit_pca(X_train_sub)
    
    # Train QSVM (will use RBF kernel by default)
    logger.info("\nTraining QSVM...")
    train_metrics = qk_model.fit(X_train_sub, y_train_sub)
    logger.info("✓ QSVM training complete")
    logger.info(f"  Support vectors: {train_metrics.get('train_support_vectors', 'N/A')}")
    
    # Evaluate on validation set (smaller subset for speed)
    logger.info("\nEvaluating on validation set (10K samples)...")
    val_subset_size = min(10000, len(X_val))
    val_idx = np.random.choice(len(X_val), val_subset_size, replace=False)
    X_val_sub = X_val[val_idx]
    y_val_sub = y_val[val_idx]
    
    val_scores, _ = qk_model.predict_scores(X_val_sub)
    val_auroc = roc_auc_score(y_val_sub, val_scores)
    val_auprc = average_precision_score(y_val_sub, val_scores)
    
    logger.info(f"Val AUROC: {val_auroc:.4f}")
    logger.info(f"Val AUPRC: {val_auprc:.4f}")
    
    # Evaluate on test set (smaller subset for speed)
    logger.info("\nEvaluating on test set (10K samples)...")
    test_subset_size = min(10000, len(X_test))
    test_idx = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_sub = X_test[test_idx]
    y_test_sub = y_test[test_idx]
    
    test_scores, _ = qk_model.predict_scores(X_test_sub)
    test_auroc = roc_auc_score(y_test_sub, test_scores)
    test_auprc = average_precision_score(y_test_sub, test_scores)
    
    logger.info(f"Test AUROC: {test_auroc:.4f}")
    logger.info(f"Test AUPRC: {test_auprc:.4f}")
    
    # Save results
    results = {
        'kernel_type': 'rbf',  # Using RBF kernel (classical)
        'training_samples': args.samples,
        'n_qubits': 8,
        'pca_components': 8,
        'support_vectors': int(train_metrics.get('train_support_vectors', 0)),
        'val_auroc': float(val_auroc),
        'val_auprc': float(val_auprc),
        'val_samples': val_subset_size,
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'test_samples': test_subset_size,
    }
    
    output_file = output_dir / "quantum_kernel_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    
    # Save model
    import pickle
    model_file = output_dir / "quantum_kernel_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(qk_model, f)
    
    logger.info(f"✓ Model saved to {model_file}")
    
    logger.info("\n" + "="*70)
    logger.info("QUANTUM KERNEL TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
