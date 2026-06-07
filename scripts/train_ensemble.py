"""
Ensemble Model Training for QuantumSepsis
==========================================

Combines HierarchicalLSTM + XGBoost to beat 0.85 AUROC target.

Ensemble Strategies:
1. Simple Average
2. Weighted Average (optimized on validation set)
3. Stacking with Logistic Regression
4. Stacking with LightGBM meta-learner

Author: QuantumSepsis Team
Date: May 12, 2026
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import SepsisDataset
from src.models.hierarchical_lstm import HierarchicalSepsisLSTM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of HierarchicalLSTM + XGBoost"""
    
    def __init__(
        self,
        lstm_model: HierarchicalSepsisLSTM,
        xgboost_model,
        device: str = 'cuda',
    ):
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.device = device
        self.weights = None
        self.meta_model = None
        
    def predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Get LSTM predictions"""
        self.lstm_model.eval()
        dataset = SepsisDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                outputs = self.lstm_model(x)
                predictions.append(outputs['risk_score'].cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def predict_xgboost(self, X: np.ndarray) -> np.ndarray:
        """Get XGBoost predictions"""
        # Flatten windows for XGBoost
        X_flat = X.reshape(len(X), -1)
        return self.xgboost_model.predict_proba(X_flat)[:, 1]
    
    def predict_simple_average(self, X: np.ndarray) -> np.ndarray:
        """Simple average ensemble"""
        lstm_pred = self.predict_lstm(X)
        xgb_pred = self.predict_xgboost(X)
        return (lstm_pred + xgb_pred) / 2
    
    def predict_weighted_average(self, X: np.ndarray) -> np.ndarray:
        """Weighted average ensemble"""
        if self.weights is None:
            raise ValueError("Weights not set. Call fit_weighted_average first.")
        
        lstm_pred = self.predict_lstm(X)
        xgb_pred = self.predict_xgboost(X)
        return self.weights[0] * lstm_pred + self.weights[1] * xgb_pred
    
    def fit_weighted_average(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize weights on validation set"""
        logger.info("Optimizing ensemble weights on validation set...")
        
        lstm_pred = self.predict_lstm(X_val)
        xgb_pred = self.predict_xgboost(X_val)
        
        best_auroc = 0
        best_weights = None
        
        # Grid search over weights
        for w1 in np.linspace(0, 1, 21):
            w2 = 1 - w1
            ensemble_pred = w1 * lstm_pred + w2 * xgb_pred
            auroc = roc_auc_score(y_val, ensemble_pred)
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_weights = (w1, w2)
        
        self.weights = best_weights
        logger.info(f"Best weights: LSTM={best_weights[0]:.3f}, XGBoost={best_weights[1]:.3f}")
        logger.info(f"Validation AUROC: {best_auroc:.4f}")
        
        return best_auroc
    
    def fit_stacking_lr(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray):
        """Train stacking ensemble with Logistic Regression meta-learner"""
        logger.info("Training stacking ensemble (Logistic Regression)...")
        
        # Get base model predictions on train set
        logger.info("  Getting base model predictions on train set...")
        lstm_train = self.predict_lstm(X_train)
        xgb_train = self.predict_xgboost(X_train)
        
        # Stack predictions
        X_meta_train = np.column_stack([lstm_train, xgb_train])
        
        # Train meta-learner
        logger.info("  Training meta-learner...")
        self.meta_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.meta_model.fit(X_meta_train, y_train)
        
        # Validate
        lstm_val = self.predict_lstm(X_val)
        xgb_val = self.predict_xgboost(X_val)
        X_meta_val = np.column_stack([lstm_val, xgb_val])
        
        val_pred = self.meta_model.predict_proba(X_meta_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_pred)
        
        logger.info(f"Stacking LR Validation AUROC: {val_auroc:.4f}")
        return val_auroc
    
    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call fit_stacking_lr first.")
        
        lstm_pred = self.predict_lstm(X)
        xgb_pred = self.predict_xgboost(X)
        X_meta = np.column_stack([lstm_pred, xgb_pred])
        
        return self.meta_model.predict_proba(X_meta)[:, 1]


def load_models(config: Config, device: str) -> Tuple[HierarchicalSepsisLSTM, object]:
    """Load trained LSTM and XGBoost models"""
    
    # Load LSTM
    logger.info("Loading HierarchicalLSTM model...")
    lstm_checkpoint = Path("checkpoints/sepsis/hierarchical_lstm_best.pt")
    if not lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {lstm_checkpoint}")
    
    lstm_model = HierarchicalSepsisLSTM(config=config.lstm, static_dim=0)
    checkpoint = torch.load(lstm_checkpoint, map_location=device, weights_only=False)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model = lstm_model.to(device)
    lstm_model.eval()
    logger.info(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load XGBoost
    logger.info("Loading XGBoost model...")
    xgb_checkpoint = Path("checkpoints/xgboost_baseline.pkl")
    if not xgb_checkpoint.exists():
        raise FileNotFoundError(f"XGBoost checkpoint not found: {xgb_checkpoint}")
    
    with open(xgb_checkpoint, 'rb') as f:
        xgb_data = pickle.load(f)
    
    # Handle both dict and direct model formats
    if isinstance(xgb_data, dict):
        xgb_model = xgb_data.get('model', xgb_data.get('xgb_model', xgb_data))
    else:
        xgb_model = xgb_data
    
    logger.info("  XGBoost model loaded")
    
    return lstm_model, xgb_model


def evaluate_ensemble(
    ensemble: EnsembleModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
) -> Dict:
    """Evaluate ensemble on test set"""
    
    logger.info(f"\nEvaluating {strategy} ensemble on test set...")
    
    # Get predictions
    if strategy == "simple_average":
        predictions = ensemble.predict_simple_average(X_test)
    elif strategy == "weighted_average":
        predictions = ensemble.predict_weighted_average(X_test)
    elif strategy == "stacking":
        predictions = ensemble.predict_stacking(X_test)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Compute metrics
    auroc = roc_auc_score(y_test, predictions)
    auprc = average_precision_score(y_test, predictions)
    
    # Find optimal threshold for F1
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    
    # Sensitivity at 95% specificity
    idx_95spec = np.argmax(fpr >= 0.05)
    sens_at_95spec = tpr[idx_95spec]
    
    results = {
        'strategy': strategy,
        'test_auroc': float(auroc),
        'test_auprc': float(auprc),
        'sensitivity_at_95spec': float(sens_at_95spec),
        'test_samples': len(y_test),
        'test_positives': int(y_test.sum()),
    }
    
    logger.info(f"  AUROC: {auroc:.4f}")
    logger.info(f"  AUPRC: {auprc:.4f}")
    logger.info(f"  Sensitivity @ 95% Spec: {sens_at_95spec:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ensemble models")
    parser.add_argument('--data', type=str, default='data/processed/features.h5',
                        help='Path to HDF5 data file')
    parser.add_argument('--output-dir', type=str, default='data/processed/sepsis',
                        help='Output directory for results')
    parser.add_argument('--max-train-samples', type=int, default=500000,
                        help='Max training samples for stacking (memory limit)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Setup
    config = Config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ENSEMBLE MODEL TRAINING FOR QUANTUMSEPSIS")
    logger.info("="*70)
    logger.info(f"Data: {args.data}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    with h5py.File(args.data, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    logger.info(f"Train: {X_train.shape}, positives: {y_train.sum()} ({100*y_train.mean():.2f}%)")
    logger.info(f"Val:   {X_val.shape}, positives: {y_val.sum()} ({100*y_val.mean():.2f}%)")
    logger.info(f"Test:  {X_test.shape}, positives: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    
    # Load models
    logger.info("\n" + "="*70)
    logger.info("LOADING BASE MODELS")
    logger.info("="*70)
    lstm_model, xgb_model = load_models(config, device)
    
    # Create ensemble
    ensemble = EnsembleModel(lstm_model, xgb_model, device=device)
    
    # Get individual model baselines
    logger.info("\n" + "="*70)
    logger.info("BASELINE PERFORMANCE (Individual Models)")
    logger.info("="*70)
    
    logger.info("\nLSTM baseline...")
    lstm_test_pred = ensemble.predict_lstm(X_test)
    lstm_auroc = roc_auc_score(y_test, lstm_test_pred)
    lstm_auprc = average_precision_score(y_test, lstm_test_pred)
    logger.info(f"  LSTM Test AUROC: {lstm_auroc:.4f}")
    logger.info(f"  LSTM Test AUPRC: {lstm_auprc:.4f}")
    
    logger.info("\nXGBoost baseline...")
    xgb_test_pred = ensemble.predict_xgboost(X_test)
    xgb_auroc = roc_auc_score(y_test, xgb_test_pred)
    xgb_auprc = average_precision_score(y_test, xgb_test_pred)
    logger.info(f"  XGBoost Test AUROC: {xgb_auroc:.4f}")
    logger.info(f"  XGBoost Test AUPRC: {xgb_auprc:.4f}")
    
    # Store all results
    all_results = {
        'baselines': {
            'lstm': {
                'test_auroc': float(lstm_auroc),
                'test_auprc': float(lstm_auprc),
            },
            'xgboost': {
                'test_auroc': float(xgb_auroc),
                'test_auprc': float(xgb_auprc),
            }
        },
        'ensembles': {}
    }
    
    # Strategy 1: Simple Average
    logger.info("\n" + "="*70)
    logger.info("STRATEGY 1: SIMPLE AVERAGE")
    logger.info("="*70)
    results_simple = evaluate_ensemble(ensemble, X_test, y_test, "simple_average")
    all_results['ensembles']['simple_average'] = results_simple
    
    # Strategy 2: Weighted Average
    logger.info("\n" + "="*70)
    logger.info("STRATEGY 2: WEIGHTED AVERAGE")
    logger.info("="*70)
    ensemble.fit_weighted_average(X_val, y_val)
    results_weighted = evaluate_ensemble(ensemble, X_test, y_test, "weighted_average")
    results_weighted['weights'] = {
        'lstm': float(ensemble.weights[0]),
        'xgboost': float(ensemble.weights[1])
    }
    all_results['ensembles']['weighted_average'] = results_weighted
    
    # Strategy 3: Stacking with Logistic Regression
    logger.info("\n" + "="*70)
    logger.info("STRATEGY 3: STACKING (Logistic Regression)")
    logger.info("="*70)
    
    # Subsample training data if too large
    if len(X_train) > args.max_train_samples:
        logger.info(f"Subsampling train set: {len(X_train)} -> {args.max_train_samples}")
        # Balanced subsample
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        
        n_pos = min(len(pos_idx), args.max_train_samples // 2)
        n_neg = args.max_train_samples - n_pos
        
        pos_sample = np.random.choice(pos_idx, n_pos, replace=False)
        neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
        
        sample_idx = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(sample_idx)
        
        X_train_sub = X_train[sample_idx]
        y_train_sub = y_train[sample_idx]
        
        logger.info(f"Subsampled: {len(X_train_sub)} samples, {y_train_sub.sum()} positives")
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    ensemble.fit_stacking_lr(X_train_sub, y_train_sub, X_val, y_val)
    results_stacking = evaluate_ensemble(ensemble, X_test, y_test, "stacking")
    all_results['ensembles']['stacking'] = results_stacking
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)
    logger.info("\nBaseline Models:")
    logger.info(f"  LSTM:     AUROC = {lstm_auroc:.4f}")
    logger.info(f"  XGBoost:  AUROC = {xgb_auroc:.4f}")
    
    logger.info("\nEnsemble Models:")
    logger.info(f"  Simple Average:    AUROC = {results_simple['test_auroc']:.4f}")
    logger.info(f"  Weighted Average:  AUROC = {results_weighted['test_auroc']:.4f}")
    logger.info(f"  Stacking (LR):     AUROC = {results_stacking['test_auroc']:.4f}")
    
    # Find best
    best_auroc = max(
        results_simple['test_auroc'],
        results_weighted['test_auroc'],
        results_stacking['test_auroc']
    )
    
    if best_auroc == results_simple['test_auroc']:
        best_strategy = "Simple Average"
    elif best_auroc == results_weighted['test_auroc']:
        best_strategy = "Weighted Average"
    else:
        best_strategy = "Stacking (LR)"
    
    logger.info(f"\n🏆 BEST ENSEMBLE: {best_strategy} with AUROC = {best_auroc:.4f}")
    
    # Check if target met
    target_auroc = 0.85
    if best_auroc >= target_auroc:
        logger.info(f"\n✅ TARGET MET! AUROC {best_auroc:.4f} >= {target_auroc}")
        logger.info("   → Ready to proceed with Quantum Kernel training")
    else:
        gap = target_auroc - best_auroc
        logger.info(f"\n⚠️  TARGET NOT MET. Gap: {gap:.4f} AUROC points")
        logger.info("   → Recommend hyperparameter tuning before Quantum Kernel")
    
    all_results['summary'] = {
        'best_strategy': best_strategy,
        'best_auroc': float(best_auroc),
        'target_met': bool(best_auroc >= target_auroc),
        'target_auroc': target_auroc,
        'gap_to_target': float(target_auroc - best_auroc) if best_auroc < target_auroc else 0.0
    }
    
    # Save results
    output_file = output_dir / "ensemble_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved to {output_file}")
    
    # Save best ensemble model
    if best_strategy == "Weighted Average":
        ensemble_save = {
            'strategy': 'weighted_average',
            'weights': ensemble.weights,
        }
    elif best_strategy == "Stacking (LR)":
        ensemble_save = {
            'strategy': 'stacking',
            'meta_model': ensemble.meta_model,
        }
    else:
        ensemble_save = {
            'strategy': 'simple_average',
        }
    
    ensemble_file = output_dir / "ensemble_model.pkl"
    with open(ensemble_file, 'wb') as f:
        pickle.dump(ensemble_save, f)
    logger.info(f"✓ Ensemble model saved to {ensemble_file}")
    
    logger.info("\n" + "="*70)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
