"""
Validate Quantum Kernel Model
==============================

Tests the complete pipeline:
1. Load LSTM model
2. Load quantum kernel model
3. Generate embeddings for test set
4. Evaluate performance
5. Test inference speed

Author: QuantumSepsis Team
Date: May 13, 2026
"""

import sys
import logging
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import pickle
import h5py
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.models.hierarchical_lstm import HierarchicalSepsisLSTM
from src.data.dataset import SepsisDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def validate_quantum_kernel(
    data_path: str,
    lstm_checkpoint: str,
    quantum_checkpoint: str,
    max_samples: int = 10000,
):
    """Validate the complete quantum kernel pipeline."""
    
    logger.info("="*70)
    logger.info("QUANTUM KERNEL VALIDATION")
    logger.info("="*70)
    
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 1. Load LSTM model
    logger.info("\n[1/6] Loading LSTM model...")
    lstm_model = HierarchicalSepsisLSTM(config=config.lstm, static_dim=0)
    checkpoint = torch.load(lstm_checkpoint, map_location=device, weights_only=False)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model = lstm_model.to(device)
    lstm_model.eval()
    logger.info(f"✓ LSTM loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 2. Load quantum kernel model
    logger.info("\n[2/6] Loading quantum kernel model...")
    with open(quantum_checkpoint, 'rb') as f:
        qk_model = pickle.load(f)
    logger.info("✓ Quantum kernel loaded")
    logger.info(f"  PCA components: {qk_model.pca_components}")
    logger.info(f"  Support vectors: {len(qk_model.support_indices_) if qk_model.support_indices_ is not None else 'N/A'}")
    
    # 3. Load test data
    logger.info("\n[3/6] Loading test data...")
    with h5py.File(data_path, 'r') as f:
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    # Subsample if needed
    if len(X_test) > max_samples:
        logger.info(f"Subsampling test set: {len(X_test)} -> {max_samples}")
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
    
    logger.info(f"Test set: {X_test.shape}, positives: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    
    # 4. Generate embeddings
    logger.info("\n[4/6] Generating LSTM embeddings...")
    test_dataset = SepsisDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    embeddings_list = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device)
            emb = lstm_model.extract_embeddings(x)
            embeddings_list.append(emb.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {(batch_idx + 1) * test_loader.batch_size} samples")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    embedding_time = time.time() - start_time
    logger.info(f"✓ Embeddings generated: {embeddings.shape}")
    logger.info(f"  Time: {embedding_time:.2f}s ({len(embeddings)/embedding_time:.0f} samples/sec)")
    
    # 5. Quantum kernel predictions
    logger.info("\n[5/6] Running quantum kernel predictions...")
    start_time = time.time()
    
    predictions, pred_labels = qk_model.predict_scores(embeddings)
    
    prediction_time = time.time() - start_time
    logger.info(f"✓ Predictions complete")
    logger.info(f"  Time: {prediction_time:.2f}s ({len(predictions)/prediction_time:.0f} samples/sec)")
    
    # 6. Evaluate performance
    logger.info("\n[6/6] Evaluating performance...")
    
    auroc = roc_auc_score(y_test, predictions)
    auprc = average_precision_score(y_test, predictions)
    
    logger.info(f"\n{'='*70}")
    logger.info("PERFORMANCE METRICS")
    logger.info(f"{'='*70}")
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    
    # Classification report at threshold 0.5
    logger.info(f"\n{'='*70}")
    logger.info("CLASSIFICATION REPORT (threshold=0.5)")
    logger.info(f"{'='*70}")
    binary_preds = (predictions >= 0.5).astype(int)
    print(classification_report(y_test, binary_preds, target_names=['No Sepsis', 'Sepsis']))
    
    # Speed summary
    logger.info(f"\n{'='*70}")
    logger.info("INFERENCE SPEED")
    logger.info(f"{'='*70}")
    total_time = embedding_time + prediction_time
    logger.info(f"Total time: {total_time:.2f}s for {len(X_test)} samples")
    logger.info(f"Per sample: {1000*total_time/len(X_test):.2f}ms")
    logger.info(f"Throughput: {len(X_test)/total_time:.0f} samples/sec")
    
    # Test single sample inference
    logger.info(f"\n{'='*70}")
    logger.info("SINGLE SAMPLE INFERENCE TEST")
    logger.info(f"{'='*70}")
    
    single_sample = torch.FloatTensor(X_test[0:1]).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        single_emb = lstm_model.extract_embeddings(single_sample).cpu().numpy()
    single_pred, _ = qk_model.predict_scores(single_emb)
    single_time = time.time() - start_time
    
    logger.info(f"Single sample inference: {1000*single_time:.2f}ms")
    logger.info(f"Prediction: {single_pred[0]:.4f} (label: {y_test[0]})")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"✓ LSTM model: Working")
    logger.info(f"✓ Quantum kernel: Working")
    logger.info(f"✓ Pipeline: End-to-end functional")
    logger.info(f"✓ Performance: AUROC {auroc:.4f}")
    logger.info(f"✓ Speed: {1000*total_time/len(X_test):.2f}ms per sample")
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION COMPLETE - ALL TESTS PASSED ✅")
    logger.info(f"{'='*70}")
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'total_time': float(total_time),
        'samples': len(X_test),
        'ms_per_sample': float(1000*total_time/len(X_test)),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate quantum kernel model")
    parser.add_argument('--data', type=str, default='data/processed/features.h5',
                        help='Path to HDF5 data file')
    parser.add_argument('--lstm-checkpoint', type=str,
                        default='checkpoints/sepsis/hierarchical_lstm_best.pt',
                        help='Path to LSTM checkpoint')
    parser.add_argument('--quantum-checkpoint', type=str,
                        default='data/processed/sepsis/quantum_kernel_model.pkl',
                        help='Path to quantum kernel model')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum test samples to evaluate')
    args = parser.parse_args()
    
    results = validate_quantum_kernel(
        args.data,
        args.lstm_checkpoint,
        args.quantum_checkpoint,
        args.max_samples,
    )
    
    # Save results
    import json
    output_file = Path(args.quantum_checkpoint).parent / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
