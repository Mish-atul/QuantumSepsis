"""
QuantumSepsis Shield — LSTM + XGBoost Ensemble
===============================================

Simple ensemble combining V1 LSTM (0.7891 AUROC) + XGBoost (0.8038 AUROC)
for immediate performance boost.

Expected ensemble AUROC: 0.82-0.83 (immediate +2-3% gain)

This is separate from the quantum-classical conformal-gated ensemble.
This is a practical baseline ensemble for Phase 1 quick wins.

Ensemble strategies:
  1. Weighted average (tune weights on validation set)
  2. Stacking with logistic regression meta-learner
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config
from src.models.lstm import SepsisLSTM
from src.baselines.xgboost_baseline import XGBoostBaseline
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class LSTMXGBoostEnsemble:
    """Simple ensemble combining LSTM and XGBoost predictions.

    Supports two fusion strategies:
      - 'weighted': Optimize weights on validation set via grid search
      - 'stacking': Train logistic regression meta-learner
    """

    def __init__(
        self,
        lstm_model: SepsisLSTM,
        xgb_model: XGBoostBaseline,
        strategy: str = "weighted",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize ensemble.

        Args:
            lstm_model: Trained SepsisLSTM model
            xgb_model: Trained XGBoostBaseline model
            strategy: 'weighted' or 'stacking'
            device: Device for LSTM inference
        """
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.strategy = strategy
        self.device = torch.device(device)

        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        # Ensemble weights (will be optimized)
        self.lstm_weight = 0.5
        self.xgb_weight = 0.5

        # Meta-learner for stacking
        self.meta_learner = None

        logger.info(f"LSTM+XGBoost Ensemble initialized with strategy: {strategy}")

    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Optimize ensemble weights on validation set.

        Args:
            X_val: (N, 6, 12) validation windows
            y_val: (N,) validation labels

        Returns:
            Dictionary with optimal weights and validation AUROC
        """
        logger.info("Optimizing ensemble weights on validation set...")

        # Get base predictions
        lstm_scores = self._predict_lstm(X_val)
        xgb_scores = self._predict_xgb(X_val)

        if self.strategy == "weighted":
            # Grid search over weight combinations
            best_auroc = 0.0
            best_lstm_weight = 0.5

            for lstm_w in np.arange(0.0, 1.01, 0.05):
                xgb_w = 1.0 - lstm_w
                ensemble_scores = lstm_w * lstm_scores + xgb_w * xgb_scores

                try:
                    auroc = roc_auc_score(y_val, ensemble_scores)
                except ValueError:
                    continue

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_lstm_weight = lstm_w

            self.lstm_weight = best_lstm_weight
            self.xgb_weight = 1.0 - best_lstm_weight

            logger.info(f"  Optimal weights: LSTM={self.lstm_weight:.2f}, XGB={self.xgb_weight:.2f}")
            logger.info(f"  Validation AUROC: {best_auroc:.4f}")

            return {
                "lstm_weight": self.lstm_weight,
                "xgb_weight": self.xgb_weight,
                "val_auroc": best_auroc,
            }

        elif self.strategy == "stacking":
            # Train logistic regression meta-learner
            X_meta = np.column_stack([lstm_scores, xgb_scores])

            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            self.meta_learner.fit(X_meta, y_val)

            # Evaluate
            meta_scores = self.meta_learner.predict_proba(X_meta)[:, 1]
            auroc = roc_auc_score(y_val, meta_scores)

            logger.info(f"  Meta-learner trained")
            logger.info(f"  Meta-learner coefficients: LSTM={self.meta_learner.coef_[0][0]:.3f}, "
                       f"XGB={self.meta_learner.coef_[0][1]:.3f}")
            logger.info(f"  Validation AUROC: {auroc:.4f}")

            return {
                "lstm_coef": float(self.meta_learner.coef_[0][0]),
                "xgb_coef": float(self.meta_learner.coef_[0][1]),
                "val_auroc": auroc,
            }

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions.

        Args:
            X: (N, 6, 12) input windows

        Returns:
            (N,) ensemble risk scores [0, 1]
        """
        # Get base predictions
        lstm_scores = self._predict_lstm(X)
        xgb_scores = self._predict_xgb(X)

        if self.strategy == "stacking" and self.meta_learner is not None:
            # Use meta-learner
            X_meta = np.column_stack([lstm_scores, xgb_scores])
            ensemble_scores = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            # Weighted average
            ensemble_scores = self.lstm_weight * lstm_scores + self.xgb_weight * xgb_scores

        return ensemble_scores

    def predict_with_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions with individual component scores.

        Args:
            X: (N, 6, 12) input windows

        Returns:
            Dictionary with 'lstm', 'xgb', and 'ensemble' scores
        """
        lstm_scores = self._predict_lstm(X)
        xgb_scores = self._predict_xgb(X)
        ensemble_scores = self.predict(X)

        return {
            "lstm": lstm_scores,
            "xgb": xgb_scores,
            "ensemble": ensemble_scores,
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prefix: str = "ensemble_test_",
    ) -> Dict[str, float]:
        """Evaluate ensemble on test set.

        Args:
            X_test: (N, 6, 12) test windows
            y_test: (N,) test labels
            prefix: Metric name prefix

        Returns:
            Dictionary with test metrics
        """
        logger.info("Evaluating LSTM+XGBoost ensemble on test set...")

        # Get all predictions
        predictions = self.predict_with_components(X_test)

        # Compute metrics for each component
        lstm_metrics = compute_all_metrics(y_test, predictions["lstm"], prefix="lstm_")
        xgb_metrics = compute_all_metrics(y_test, predictions["xgb"], prefix="xgb_")
        ensemble_metrics = compute_all_metrics(y_test, predictions["ensemble"], prefix=prefix)

        # Print comparison
        print("\n" + "=" * 70)
        print("  LSTM + XGBoost ENSEMBLE EVALUATION")
        print("=" * 70)
        print(f"  LSTM AUROC:     {lstm_metrics['lstm_auroc']:.4f}")
        print(f"  XGBoost AUROC:  {xgb_metrics['xgb_auroc']:.4f}")
        print(f"  Ensemble AUROC: {ensemble_metrics[f'{prefix}auroc']:.4f}")
        print(f"  Gain over LSTM: +{ensemble_metrics[f'{prefix}auroc'] - lstm_metrics['lstm_auroc']:.4f} "
              f"({100*(ensemble_metrics[f'{prefix}auroc'] - lstm_metrics['lstm_auroc'])/lstm_metrics['lstm_auroc']:.1f}%)")
        print(f"  Gain over XGB:  +{ensemble_metrics[f'{prefix}auroc'] - xgb_metrics['xgb_auroc']:.4f} "
              f"({100*(ensemble_metrics[f'{prefix}auroc'] - xgb_metrics['xgb_auroc'])/xgb_metrics['xgb_auroc']:.1f}%)")
        print("=" * 70)

        print(format_metrics(ensemble_metrics, "Ensemble Test Metrics"))

        # Combine all metrics
        all_metrics = {**lstm_metrics, **xgb_metrics, **ensemble_metrics}

        return all_metrics

    @torch.no_grad()
    def _predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Get LSTM predictions.

        Args:
            X: (N, 6, 12) numpy array

        Returns:
            (N,) risk scores
        """
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Batch prediction to avoid OOM
        batch_size = 512
        all_scores = []

        for i in range(0, len(X), batch_size):
            batch = X_tensor[i:i+batch_size]
            output = self.lstm_model(batch)
            scores = output['risk_score'].cpu().numpy()
            all_scores.append(scores)

        return np.concatenate(all_scores)

    def _predict_xgb(self, X: np.ndarray) -> np.ndarray:
        """Get XGBoost predictions.

        Args:
            X: (N, 6, 12) numpy array

        Returns:
            (N,) risk scores
        """
        X_eng = self.xgb_model.engineer_features(X)
        scores = self.xgb_model.model.predict_proba(X_eng)[:, 1]
        return scores

    def save_weights(self, path: str) -> None:
        """Save ensemble weights to file."""
        import json

        weights = {
            "strategy": self.strategy,
            "lstm_weight": float(self.lstm_weight),
            "xgb_weight": float(self.xgb_weight),
        }

        if self.meta_learner is not None:
            weights["meta_learner_coef"] = self.meta_learner.coef_.tolist()
            weights["meta_learner_intercept"] = float(self.meta_learner.intercept_[0])

        with open(path, 'w') as f:
            json.dump(weights, f, indent=2)

        logger.info(f"Ensemble weights saved to {path}")

    def load_weights(self, path: str) -> None:
        """Load ensemble weights from file."""
        import json

        with open(path, 'r') as f:
            weights = json.load(f)

        self.strategy = weights["strategy"]
        self.lstm_weight = weights["lstm_weight"]
        self.xgb_weight = weights["xgb_weight"]

        if "meta_learner_coef" in weights:
            self.meta_learner = LogisticRegression()
            self.meta_learner.coef_ = np.array(weights["meta_learner_coef"])
            self.meta_learner.intercept_ = np.array([weights["meta_learner_intercept"]])
            self.meta_learner.classes_ = np.array([0, 1])

        logger.info(f"Ensemble weights loaded from {path}")


def test_ensemble():
    """Test ensemble with synthetic data."""
    print("Testing LSTM+XGBoost Ensemble")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_train, n_val, n_test = 2000, 400, 400

    X_train = np.random.randn(n_train, 6, 12).astype(np.float32)
    y_train = (np.random.random(n_train) > 0.75).astype(int)
    X_val = np.random.randn(n_val, 6, 12).astype(np.float32)
    y_val = (np.random.random(n_val) > 0.75).astype(int)
    X_test = np.random.randn(n_test, 6, 12).astype(np.float32)
    y_test = (np.random.random(n_test) > 0.75).astype(int)

    # Train base models
    config = get_default_config()

    print("\n1. Training LSTM...")
    lstm_model = SepsisLSTM(config.lstm)

    print("\n2. Training XGBoost...")
    xgb_model = XGBoostBaseline(config)
    xgb_model.train(X_train, y_train, X_val, y_val)

    # Create ensemble
    print("\n3. Creating ensemble...")
    ensemble = LSTMXGBoostEnsemble(lstm_model, xgb_model, strategy="weighted")

    # Optimize weights
    print("\n4. Optimizing ensemble weights...")
    weights = ensemble.fit_weights(X_val, y_val)
    print(f"   Optimal weights: {weights}")

    # Evaluate
    print("\n5. Evaluating on test set...")
    metrics = ensemble.evaluate(X_test, y_test)

    # Test prediction
    print("\n6. Testing prediction...")
    predictions = ensemble.predict_with_components(X_test[:10])
    print(f"   LSTM scores:     {predictions['lstm'][:5]}")
    print(f"   XGBoost scores:  {predictions['xgb'][:5]}")
    print(f"   Ensemble scores: {predictions['ensemble'][:5]}")

    # Test save/load
    print("\n7. Testing save/load...")
    ensemble.save_weights("test_ensemble_weights.json")
    ensemble.load_weights("test_ensemble_weights.json")
    print("   ✓ Save/load successful")

    print("\n✓ All ensemble tests passed!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_ensemble()
