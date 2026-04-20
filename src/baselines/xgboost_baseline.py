"""
QuantumSepsis Shield — XGBoost Baseline
=========================================

Baseline comparison using XGBoost on flattened features.
Represents the "strong classical baseline" that our quantum-enhanced
system must outperform.

Input: Flattened 6×12 = 72 features
Expected AUROC: 0.78 – 0.82
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """XGBoost baseline for sepsis prediction.
    
    Uses raw flattened features (6 hours × 12 features = 72 features)
    with additional engineered features (trends, min, max, mean).
    """
    
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_default_config()
        
        self.config = config
        self.model = None
    
    def engineer_features(self, X: np.ndarray) -> np.ndarray:
        """Engineer additional features from the window tensor.
        
        Input: (N, 6, 12)
        Output: (N, 72 + 48 + 12) = (N, 132) features
        
        Additional features per variable (12 × 4 = 48):
            - Mean over 6 hours
            - Std over 6 hours
            - Linear trend (slope)
            - Last-minus-first (delta)
        
        Plus 12 latest values.
        """
        N, T, F = X.shape
        
        # Flattened features: 72
        flat = X.reshape(N, -1)
        
        # Statistical features: 48
        means = np.nanmean(X, axis=1)      # (N, 12)
        stds = np.nanstd(X, axis=1)        # (N, 12)
        
        # Trend (slope via linear regression per variable)
        trends = np.zeros((N, F))
        t = np.arange(T)
        for f in range(F):
            for i in range(N):
                vals = X[i, :, f]
                valid = ~np.isnan(vals)
                if valid.sum() >= 2:
                    slope = np.polyfit(t[valid], vals[valid], 1)[0]
                    trends[i, f] = slope
        
        # Delta (last - first)
        deltas = X[:, -1, :] - X[:, 0, :]
        
        # Latest values: 12
        latest = X[:, -1, :]
        
        # Combine all features
        engineered = np.hstack([flat, means, stds, trends, deltas, latest])
        
        # Replace NaN with 0
        engineered = np.nan_to_num(engineered, 0.0)
        
        return engineered
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Train XGBoost model.
        
        Args:
            X_train: (N, 6, 12) training windows
            y_train: (N,) labels
            X_val: (M, 6, 12) validation windows
            y_val: (M,) labels
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training XGBoost baseline...")
        
        # Engineer features
        X_train_eng = self.engineer_features(X_train)
        X_val_eng = self.engineer_features(X_val)
        
        logger.info(f"  Engineered features: {X_train_eng.shape[1]}")
        
        # Compute scale_pos_weight for imbalanced classes
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = 10.0  # FN = 10× FP to match our asymmetric loss
        
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
        
        self.model.fit(
            X_train_eng, y_train,
            eval_set=[(X_val_eng, y_val)],
            verbose=False,
        )
        
        # Evaluate
        val_scores = self.model.predict_proba(X_val_eng)[:, 1]
        val_metrics = compute_all_metrics(y_val, val_scores, prefix="val_")
        
        logger.info(f"  Val AUROC: {val_metrics['val_auroc']:.4f}")
        logger.info(f"  Val AUPRC: {val_metrics['val_auprc']:.4f}")
        
        return val_metrics
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on test set.
        
        Args:
            X_test: (N, 6, 12) test windows
            y_test: (N,) labels
        
        Returns:
            Test metrics
        """
        assert self.model is not None, "Must train first"
        
        X_test_eng = self.engineer_features(X_test)
        test_scores = self.model.predict_proba(X_test_eng)[:, 1]
        
        metrics = compute_all_metrics(y_test, test_scores, prefix="xgb_test_")
        print(format_metrics(metrics, "XGBoost Baseline — Test Results"))
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top feature importances."""
        assert self.model is not None
        
        importances = self.model.feature_importances_
        feature_names = self._get_feature_names()
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        result = {}
        for idx in indices:
            if idx < len(feature_names):
                result[feature_names[idx]] = float(importances[idx])
        
        return result
    
    def _get_feature_names(self):
        """Generate feature names for all engineered features."""
        feat_names = self.config.data.feature_names
        names = []
        
        # Flattened: hour_0_hr, hour_0_sbp, ...
        for h in range(6):
            for f in feat_names:
                names.append(f"h{h}_{f}")
        
        # Statistical features
        for suffix in ["mean", "std", "trend", "delta"]:
            for f in feat_names:
                names.append(f"{f}_{suffix}")
        
        # Latest
        for f in feat_names:
            names.append(f"{f}_latest")
        
        return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("XGBoost Baseline — Synthetic Test")
    print("=" * 60)
    
    np.random.seed(42)
    n_train, n_val, n_test = 2000, 400, 400
    
    X_train = np.random.randn(n_train, 6, 12).astype(np.float32)
    y_train = (np.random.random(n_train) > 0.75).astype(int)
    X_val = np.random.randn(n_val, 6, 12).astype(np.float32)
    y_val = (np.random.random(n_val) > 0.75).astype(int)
    X_test = np.random.randn(n_test, 6, 12).astype(np.float32)
    y_test = (np.random.random(n_test) > 0.75).astype(int)
    
    baseline = XGBoostBaseline()
    val_metrics = baseline.train(X_train, y_train, X_val, y_val)
    test_metrics = baseline.evaluate(X_test, y_test)
    
    print("\nTop feature importances:")
    importances = baseline.get_feature_importance(10)
    for name, imp in importances.items():
        print(f"  {name:25s}: {imp:.4f}")
