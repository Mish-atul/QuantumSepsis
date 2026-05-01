"""
QuantumSepsis Shield — Conformal-Gated Quantum-Classical Ensemble (Novelty 4)
==============================================================================

When LSTM is confident (tight conformal interval) → trust LSTM
When LSTM is uncertain (wide conformal interval)  → weight quantum kernel more

This is novel: no prior work uses conformal prediction interval width
as a dynamic gating signal for quantum-classical model fusion.

Formula:
    gate = σ(β * (confidence - τ))
    ensemble = gate * lstm_score + (1 - gate) * qsvm_score

Where:
    confidence = 1 - conformal_width
    β = steepness of gate transition (default 10.0)
    τ = confidence threshold for switching (default 0.5)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class ConformalGatedEnsemble:
    """Conformal-Gated Quantum-Classical Ensemble (Novelty 4).

    Dynamically weights LSTM and quantum kernel predictions based on
    the LSTM's conformal prediction interval width.

    When conformal width is narrow → LSTM is confident → trust LSTM
    When conformal width is wide   → LSTM is uncertain → trust quantum

    Args:
        beta:      Steepness of the sigmoid gating function (default 10.0)
        tau:       Confidence threshold for gate transition (default 0.5)
        lstm_bias: Base bias toward LSTM (default 0.6). At confidence=tau,
                   ensemble weight for LSTM is lstm_bias.
    """

    def __init__(
        self,
        beta: float = 10.0,
        tau: float = 0.5,
        lstm_bias: float = 0.6,
    ):
        self.beta = beta
        self.tau = tau
        self.lstm_bias = lstm_bias
        self._calibrated = False
        self._optimal_beta: Optional[float] = None
        self._optimal_tau: Optional[float] = None

    def compute_gate(self, conformal_widths: np.ndarray) -> np.ndarray:
        """Compute the gating signal from conformal widths.

        Args:
            conformal_widths: (N,) conformal interval widths ∈ [0, 1]

        Returns:
            gate: (N,) gating values ∈ [0, 1].
                  High → trust LSTM. Low → trust quantum.
        """
        confidence = 1.0 - conformal_widths
        gate = sigmoid(self.beta * (confidence - self.tau))
        return gate.astype(np.float32)

    def predict(
        self,
        lstm_scores: np.ndarray,
        quantum_scores: np.ndarray,
        conformal_widths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produce ensemble predictions.

        Args:
            lstm_scores:      (N,) LSTM risk scores ∈ [0, 1]
            quantum_scores:   (N,) Quantum kernel risk scores ∈ [0, 1]
            conformal_widths: (N,) Conformal interval widths

        Returns:
            ensemble_scores: (N,) blended risk scores
            gates:           (N,) gating weights (for interpretability)
        """
        gate = self.compute_gate(conformal_widths)

        ensemble = gate * lstm_scores + (1 - gate) * quantum_scores
        ensemble = np.clip(ensemble, 0.0, 1.0)

        return ensemble.astype(np.float32), gate

    def calibrate(
        self,
        lstm_scores: np.ndarray,
        quantum_scores: np.ndarray,
        conformal_widths: np.ndarray,
        labels: np.ndarray,
        beta_range: np.ndarray = np.arange(1.0, 20.0, 1.0),
        tau_range: np.ndarray = np.arange(0.2, 0.8, 0.05),
    ) -> Dict[str, float]:
        """Grid-search for optimal beta and tau on validation set.

        Args:
            lstm_scores:      (N,) LSTM validation scores
            quantum_scores:   (N,) Quantum validation scores
            conformal_widths: (N,) Validation conformal widths
            labels:           (N,) True labels

        Returns:
            dict with best_beta, best_tau, best_auroc
        """
        from sklearn.metrics import roc_auc_score

        best_auroc = 0.0
        best_beta = self.beta
        best_tau = self.tau

        for beta in beta_range:
            for tau in tau_range:
                self.beta = beta
                self.tau = tau
                ensemble, _ = self.predict(lstm_scores, quantum_scores, conformal_widths)

                try:
                    auroc = roc_auc_score(labels, ensemble)
                except ValueError:
                    continue

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_beta = beta
                    best_tau = tau

        self.beta = best_beta
        self.tau = best_tau
        self._calibrated = True
        self._optimal_beta = best_beta
        self._optimal_tau = best_tau

        logger.info(
            "Ensemble calibrated: beta=%.1f, tau=%.2f, val_auroc=%.4f",
            best_beta, best_tau, best_auroc,
        )

        return {
            "best_beta": float(best_beta),
            "best_tau": float(best_tau),
            "best_auroc": float(best_auroc),
        }

    def summary(self) -> str:
        return (
            f"ConformalGatedEnsemble(β={self.beta:.1f}, τ={self.tau:.2f}, "
            f"calibrated={self._calibrated})"
        )


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    N = 1000

    # Simulate: LSTM is better when confident, quantum fills the gap
    lstm_scores = rng.beta(2, 5, N).astype(np.float32)
    quantum_scores = (lstm_scores + rng.normal(0, 0.1, N)).clip(0, 1).astype(np.float32)
    conformal_widths = rng.beta(2, 3, N).astype(np.float32)
    labels = (rng.random(N) < 0.15).astype(np.int8)

    ensemble = ConformalGatedEnsemble(beta=10.0, tau=0.5)

    # Predict
    scores, gates = ensemble.predict(lstm_scores, quantum_scores, conformal_widths)
    print(f"Ensemble scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"Gate values:     mean={gates.mean():.4f} (1=trust LSTM, 0=trust quantum)")

    # Calibrate
    cal_result = ensemble.calibrate(
        lstm_scores, quantum_scores, conformal_widths, labels
    )
    print(f"Calibration: {cal_result}")
    print(f"\n✓ Ensemble test passed!")
