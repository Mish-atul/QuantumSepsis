"""
QuantumSepsis Shield — Real Ensemble Evaluation
================================================

Evaluates LSTM + XGBoost ensemble on real MIMIC-IV test data.

Usage:
    python evaluate_real_ensemble.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config, get_default_config
from src.models.lstm import SepsisLSTM
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_all_metrics, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RealEnsembleEvaluator:
    """Evaluates LSTM + XGBoost ensemble on real data."""

    def __init__(
        self,
        lstm_checkpoint: str,
        xgb_checkpoint: str,
        data_path: str,
        config: Config,
    ):
        self.lstm_checkpoint = lstm_checkpoint
        self.xgb_checkpoint = xgb_checkpoint
        self.data_path = data_path
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_lstm_model(self) -> SepsisLSTM:
        """Load trained LSTM model."""
        logger.info(f"Loading LSTM model from {self.lstm_checkpoint}")
        
        checkpoint = torch.load(self.lstm_checkpoint, map_location=self.device, weights_only=False)
        
        model = SepsisLSTM(self.config.lstm).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(f"LSTM loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return model

    def load_xgb_model(self):
        """Load trained XGBoost model."""
        logger.info(f"Loading XGBoost model from {self.xgb_checkpoint}")
        
        with open(self.xgb_checkpoint, 'rb') as f:
            xgb_data = pickle.load(f)
        
        xgb_model = xgb_data['model']
        logger.info("XGBoost model loaded")
        return xgb_model

    @torch.no_grad()
    def get_lstm_predictions(self, model: SepsisLSTM, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Get LSTM predictions on dataset."""
        logger.info("Getting LSTM predictions...")
        
        all_scores = []
        all_labels = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = model(batch_x)
            all_scores.append(output['risk_score'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_pred_lstm = np.concatenate(all_scores)
        
        logger.info(f"LSTM predictions: {len(y_true)} samples")
        return y_true, y_pred_lstm

    def get_xgb_predictions(self, model, dataloader) -> np.ndarray:
        """Get XGBoost predictions on dataset."""
        logger.info("Getting XGBoost predictions...")
        
        # Import XGBoost baseline for feature engineering
        from src.baselines.xgboost_baseline import XGBoostBaseline
        xgb_baseline = XGBoostBaseline(self.config)
        
        all_scores = []
        
        for batch_x, batch_y in dataloader:
            # Engineer features using XGBoost's method: (batch, 6, 12) -> (batch, 132)
            batch_x_eng = xgb_baseline.engineer_features(batch_x.numpy())
            
            # XGBoost predict_proba returns [prob_class_0, prob_class_1]
            proba = model.predict_proba(batch_x_eng)
            all_scores.append(proba[:, 1])  # Probability of positive class
        
        y_pred_xgb = np.concatenate(all_scores)
        
        logger.info(f"XGBoost predictions: {len(y_pred_xgb)} samples")
        return y_pred_xgb

    def optimize_ensemble_weights(
        self,
        y_true: np.ndarray,
        y_pred_lstm: np.ndarray,
        y_pred_xgb: np.ndarray,
    ) -> Tuple[float, float]:
        """Find optimal ensemble weights on validation set."""
        from sklearn.metrics import roc_auc_score
        
        logger.info("Optimizing ensemble weights...")
        
        best_auroc = 0.0
        best_lstm_weight = 0.0
        best_xgb_weight = 1.0
        
        # Grid search over weights
        for lstm_w in np.linspace(0, 1, 21):
            xgb_w = 1.0 - lstm_w
            
            y_pred_ensemble = lstm_w * y_pred_lstm + xgb_w * y_pred_xgb
            auroc = roc_auc_score(y_true, y_pred_ensemble)
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_lstm_weight = lstm_w
                best_xgb_weight = xgb_w
        
        logger.info(f"Optimal weights: LSTM={best_lstm_weight:.3f}, XGBoost={best_xgb_weight:.3f}")
        logger.info(f"Validation AUROC: {best_auroc:.4f}")
        
        return best_lstm_weight, best_xgb_weight

    def evaluate(self) -> Dict[str, float]:
        """Run full ensemble evaluation."""
        logger.info("\n" + "=" * 70)
        logger.info("REAL ENSEMBLE EVALUATION - LSTM + XGBoost")
        logger.info("=" * 70)
        
        # Load data
        logger.info("\n1. Loading data...")
        train_loader, val_loader, test_loader = create_dataloaders(
            self.data_path, self.config
        )
        
        # Load models
        logger.info("\n2. Loading models...")
        lstm_model = self.load_lstm_model()
        xgb_model = self.load_xgb_model()
        
        # Get validation predictions for weight optimization
        logger.info("\n3. Getting validation predictions...")
        y_val, y_val_lstm = self.get_lstm_predictions(lstm_model, val_loader)
        y_val_xgb = self.get_xgb_predictions(xgb_model, val_loader)
        
        # Optimize weights
        logger.info("\n4. Optimizing ensemble weights...")
        lstm_weight, xgb_weight = self.optimize_ensemble_weights(
            y_val, y_val_lstm, y_val_xgb
        )
        
        # Get test predictions
        logger.info("\n5. Getting test predictions...")
        y_test, y_test_lstm = self.get_lstm_predictions(lstm_model, test_loader)
        y_test_xgb = self.get_xgb_predictions(xgb_model, test_loader)
        
        # Compute ensemble predictions
        logger.info("\n6. Computing ensemble predictions...")
        y_test_ensemble = lstm_weight * y_test_lstm + xgb_weight * y_test_xgb
        
        # Evaluate all models
        logger.info("\n7. Evaluating models...")
        lstm_metrics = compute_all_metrics(y_test, y_test_lstm, prefix="lstm_")
        xgb_metrics = compute_all_metrics(y_test, y_test_xgb, prefix="xgb_")
        ensemble_metrics = compute_all_metrics(y_test, y_test_ensemble, prefix="ensemble_")
        
        # Print results
        logger.info("\n" + "=" * 70)
        logger.info("ENSEMBLE EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"LSTM Test AUROC:     {lstm_metrics['lstm_auroc']:.4f}")
        logger.info(f"XGBoost Test AUROC:  {xgb_metrics['xgb_auroc']:.4f}")
        logger.info(f"Ensemble Test AUROC: {ensemble_metrics['ensemble_auroc']:.4f}")
        logger.info("")
        logger.info(f"Gain over LSTM:      +{ensemble_metrics['ensemble_auroc'] - lstm_metrics['lstm_auroc']:.4f} "
                   f"({100*(ensemble_metrics['ensemble_auroc'] - lstm_metrics['lstm_auroc'])/lstm_metrics['lstm_auroc']:.1f}%)")
        logger.info(f"Gain over XGBoost:   +{ensemble_metrics['ensemble_auroc'] - xgb_metrics['xgb_auroc']:.4f} "
                   f"({100*(ensemble_metrics['ensemble_auroc'] - xgb_metrics['xgb_auroc'])/xgb_metrics['xgb_auroc']:.1f}%)")
        logger.info("=" * 70)
        
        # Detailed metrics
        print("\n" + format_metrics(lstm_metrics, "LSTM Test Metrics"))
        print("\n" + format_metrics(xgb_metrics, "XGBoost Test Metrics"))
        print("\n" + format_metrics(ensemble_metrics, "Ensemble Test Metrics"))
        
        # Combine all results
        results = {
            "lstm_weight": float(lstm_weight),
            "xgb_weight": float(xgb_weight),
            "val_auroc": float(self.optimize_ensemble_weights(y_val, y_val_lstm, y_val_xgb)[0]),
            **lstm_metrics,
            **xgb_metrics,
            **ensemble_metrics,
        }
        
        return results


def main():
    """Main evaluation function."""
    # Paths
    lstm_checkpoint = "checkpoints/lstm_v1_improved_best.pt"
    xgb_checkpoint = "checkpoints/xgboost_baseline.pkl"
    data_path = "data/processed/features.h5"
    output_path = "data/processed/real_ensemble_results.json"
    
    # Check files exist
    if not Path(lstm_checkpoint).exists():
        logger.error(f"LSTM checkpoint not found: {lstm_checkpoint}")
        logger.info("Available checkpoints:")
        for f in Path("checkpoints").glob("*.pt"):
            logger.info(f"  - {f}")
        return
    
    if not Path(xgb_checkpoint).exists():
        logger.error(f"XGBoost checkpoint not found: {xgb_checkpoint}")
        logger.info("You may need to train XGBoost baseline first:")
        logger.info("  python -m src.baselines.xgboost_baseline --data data/processed/features.h5")
        return
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Load config
    config = get_default_config()
    
    # Run evaluation
    evaluator = RealEnsembleEvaluator(
        lstm_checkpoint=lstm_checkpoint,
        xgb_checkpoint=xgb_checkpoint,
        data_path=data_path,
        config=config,
    )
    
    results = evaluator.evaluate()
    
    # Save results
    logger.info(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✓ Real ensemble evaluation complete!")
    logger.info(f"\nFinal AUROC: {results['ensemble_auroc']:.4f}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"V1 Baseline:         0.7891")
    logger.info(f"V1 Improved (Phase1): {results['lstm_auroc']:.4f}")
    logger.info(f"XGBoost Baseline:    {results['xgb_auroc']:.4f}")
    logger.info(f"Ensemble (FINAL):    {results['ensemble_auroc']:.4f}")
    logger.info("")
    
    if results['ensemble_auroc'] >= 0.85:
        logger.info("🎉 TARGET ACHIEVED! AUROC ≥ 0.85")
    elif results['ensemble_auroc'] >= 0.82:
        logger.info("✅ Good progress! Consider Phase 2 for final push to 0.85")
    elif results['ensemble_auroc'] >= 0.80:
        logger.info("✅ Solid improvement! Phase 2 recommended to reach 0.85")
    else:
        logger.info("⚠️  Below 0.80. Consider enhanced features + Phase 2")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
