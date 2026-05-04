"""
Train XGBoost Baseline on Real MIMIC-IV Data
=============================================

Usage:
    python train_xgboost_real.py --data data/processed/features.h5
"""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np

from src.baselines.xgboost_baseline import XGBoostBaseline
from src.config import get_default_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: str):
    """Load data from HDF5 file."""
    logger.info(f"Loading data from {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    logger.info(f"Train: {X_train.shape}, {y_train.sum()} positive")
    logger.info(f"Val:   {X_val.shape}, {y_val.sum()} positive")
    logger.info(f"Test:  {X_test.shape}, {y_test.sum()} positive")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to features.h5")
    parser.add_argument("--output", type=str, default="checkpoints/xgboost_baseline.pkl",
                       help="Output path for trained model")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data)
    
    # Train model
    logger.info("\nTraining XGBoost baseline...")
    config = get_default_config()
    baseline = XGBoostBaseline(config)
    
    val_metrics = baseline.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = baseline.evaluate(X_test, y_test)
    
    # Feature importance
    logger.info("\nTop 20 feature importances:")
    importances = baseline.get_feature_importance(20)
    for name, imp in importances.items():
        logger.info(f"  {name:30s}: {imp:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to {args.output}")
    model_data = {
        'model': baseline.model,
        'config': config,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importances': importances,
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("\n✓ XGBoost baseline training complete!")
    logger.info(f"Test AUROC: {test_metrics['xgb_test_auroc']:.4f}")


if __name__ == "__main__":
    main()
