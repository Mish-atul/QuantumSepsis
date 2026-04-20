"""
QuantumSepsis Shield — Outcome Learning Agent
===============================================

Post-case analysis and adaptive threshold tuning.

After a case resolves (72 hours post-alert), the Outcome Learning Agent:
  1. Compares predicted risk with actual outcome
  2. Identifies false negatives and near-misses
  3. Adaptively updates risk thresholds per ICU unit
  4. Doubles asymmetric penalty for patient profiles that triggered safety overrides

This implements Novelty 2: Adversarial Tripwire-Gated Asymmetric Safety
with feedback learning.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_default_config

logger = logging.getLogger(__name__)


@dataclass
class CaseOutcome:
    """Outcome record for a single patient assessment cycle."""
    stay_id: int
    icu_unit: str
    prediction_time: str
    risk_score: float
    confidence: float
    alert_level: str          # What was triggered
    red_team_triggered: bool
    n_tripwires: int
    actual_sepsis: bool       # Ground truth (72h later)
    time_to_onset_hours: Optional[float] = None  # If sepsis occurred
    
    @property
    def was_false_negative(self) -> bool:
        """Was this a false negative?"""
        return self.actual_sepsis and self.alert_level == "WATCH"
    
    @property
    def was_false_positive(self) -> bool:
        """Was this a false positive?"""
        return not self.actual_sepsis and self.alert_level in ["AMBER", "CRITICAL"]
    
    @property
    def was_near_miss(self) -> bool:
        """Was this a near-miss (Red Team caught what model missed)?"""
        return self.actual_sepsis and self.risk_score < 0.3 and self.red_team_triggered


@dataclass
class ThresholdState:
    """Adaptive threshold state for an ICU unit."""
    watch_threshold: float = 0.3
    amber_threshold: float = 0.6
    
    # Tracking
    total_cases: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    near_misses: int = 0
    
    # Adaptive loss multiplier
    loss_multiplier: float = 1.0


class OutcomeLearningAgent:
    """Adaptive learning agent for post-case analysis and threshold tuning.
    
    Implements the feedback loop:
        Near-miss event → double penalty → retrain quantum kernel → better predictions
    
    Update rule:
        threshold_new = threshold_old + η × (target_sensitivity - observed_sensitivity)
        η = 0.05 (conservative learning rate)
    
    Args:
        learning_rate: How quickly thresholds adapt (default: 0.05)
        target_sensitivity: Target sensitivity level (default: 0.95)
        min_cases_for_update: Minimum resolved cases before updating (default: 20)
        update_frequency_hours: How often to update thresholds (default: 168 = 1 week)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        target_sensitivity: float = 0.95,
        min_cases_for_update: int = 20,
        update_frequency_hours: int = 168,
    ):
        self.learning_rate = learning_rate
        self.target_sensitivity = target_sensitivity
        self.min_cases_for_update = min_cases_for_update
        self.update_frequency_hours = update_frequency_hours
        
        # Per-unit threshold states
        self.unit_states: Dict[str, ThresholdState] = defaultdict(ThresholdState)
        
        # Case history
        self.case_history: List[CaseOutcome] = []
        
        # Near-miss profiles for adaptive loss
        self.near_miss_profiles: List[Dict] = []
    
    def record_outcome(self, outcome: CaseOutcome) -> Dict[str, str]:
        """Record a case outcome and check for near-misses.
        
        Args:
            outcome: Resolved case outcome
        
        Returns:
            Dictionary with feedback actions
        """
        self.case_history.append(outcome)
        
        unit = outcome.icu_unit
        state = self.unit_states[unit]
        state.total_cases += 1
        
        feedback = {}
        
        if outcome.actual_sepsis and outcome.alert_level in ["AMBER", "CRITICAL"]:
            state.true_positives += 1
        
        if outcome.was_false_negative:
            state.false_negatives += 1
            feedback["alert"] = (
                f"FALSE NEGATIVE detected in {unit}: "
                f"Risk={outcome.risk_score:.2f}, Actual=sepsis"
            )
            logger.warning(feedback["alert"])
        
        if outcome.was_false_positive:
            state.false_positives += 1
        
        if outcome.was_near_miss:
            state.near_misses += 1
            feedback["near_miss"] = (
                f"NEAR-MISS in {unit}: Red Team caught sepsis "
                f"(risk={outcome.risk_score:.2f}, {outcome.n_tripwires} tripwires). "
                f"Loss multiplier doubled."
            )
            logger.warning(feedback["near_miss"])
            
            # NOVELTY 2: Double the loss multiplier for this patient profile
            state.loss_multiplier *= 2.0
            state.loss_multiplier = min(state.loss_multiplier, 32.0)  # Cap at 32×
            
            # Record profile for retraining
            self.near_miss_profiles.append({
                "stay_id": outcome.stay_id,
                "icu_unit": unit,
                "risk_score": outcome.risk_score,
                "loss_multiplier": state.loss_multiplier,
            })
            
            feedback["loss_update"] = (
                f"Loss multiplier for {unit}: {state.loss_multiplier:.0f}×"
            )
        
        # Check if we should update thresholds
        if state.total_cases >= self.min_cases_for_update:
            threshold_update = self._update_thresholds(unit)
            if threshold_update:
                feedback["threshold_update"] = threshold_update
        
        return feedback
    
    def _update_thresholds(self, unit: str) -> Optional[str]:
        """Adaptively update risk thresholds for an ICU unit.
        
        Update rule:
            threshold_new = threshold_old + η × (target - observed)
        
        If observed sensitivity is below target, lower thresholds
        (more aggressive alerting).
        """
        state = self.unit_states[unit]
        
        total_positive = state.true_positives + state.false_negatives
        if total_positive == 0:
            return None
        
        observed_sensitivity = state.true_positives / total_positive
        
        if abs(observed_sensitivity - self.target_sensitivity) < 0.01:
            return None  # Close enough
        
        # Compute threshold adjustment
        delta = self.learning_rate * (self.target_sensitivity - observed_sensitivity)
        
        old_watch = state.watch_threshold
        old_amber = state.amber_threshold
        
        # Lower thresholds if sensitivity too low (be more aggressive)
        state.watch_threshold = max(0.05, state.watch_threshold - delta)
        state.amber_threshold = max(0.2, state.amber_threshold - delta)
        
        # Ensure ordering
        state.watch_threshold = min(state.watch_threshold, state.amber_threshold - 0.1)
        
        msg = (
            f"Threshold update for {unit}: "
            f"observed_sensitivity={observed_sensitivity:.2%} "
            f"(target={self.target_sensitivity:.2%}). "
            f"WATCH: {old_watch:.2f}→{state.watch_threshold:.2f}, "
            f"AMBER: {old_amber:.2f}→{state.amber_threshold:.2f}"
        )
        logger.info(msg)
        
        return msg
    
    def get_unit_thresholds(self, unit: str) -> Dict[str, float]:
        """Get current thresholds for an ICU unit."""
        state = self.unit_states[unit]
        return {
            "watch_threshold": state.watch_threshold,
            "amber_threshold": state.amber_threshold,
            "loss_multiplier": state.loss_multiplier,
            "total_cases": state.total_cases,
            "sensitivity": (
                state.true_positives / max(state.true_positives + state.false_negatives, 1)
            ),
        }
    
    def get_near_miss_weights(self) -> Dict[int, float]:
        """Get per-stay loss multipliers for retraining.
        
        Returns:
            Dictionary mapping stay_id → loss_multiplier
        """
        weights = {}
        for profile in self.near_miss_profiles:
            sid = profile["stay_id"]
            weights[sid] = max(weights.get(sid, 1.0), profile["loss_multiplier"])
        return weights
    
    def summary(self) -> str:
        """Return summary of all unit states."""
        lines = ["Outcome Learning Agent Summary:", "=" * 50]
        
        for unit, state in sorted(self.unit_states.items()):
            total_pos = state.true_positives + state.false_negatives
            sensitivity = state.true_positives / max(total_pos, 1)
            
            lines.append(f"\n  {unit}:")
            lines.append(f"    Total cases:     {state.total_cases}")
            lines.append(f"    True positives:  {state.true_positives}")
            lines.append(f"    False negatives: {state.false_negatives}")
            lines.append(f"    False positives: {state.false_positives}")
            lines.append(f"    Near-misses:     {state.near_misses}")
            lines.append(f"    Sensitivity:     {sensitivity:.2%}")
            lines.append(f"    Loss multiplier: {state.loss_multiplier:.0f}×")
            lines.append(f"    Thresholds:      WATCH={state.watch_threshold:.2f}, AMBER={state.amber_threshold:.2f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = OutcomeLearningAgent(min_cases_for_update=5)
    
    print("Outcome Learning Agent — Test")
    print("=" * 60)
    
    # Simulate a series of outcomes
    np.random.seed(42)
    
    for i in range(30):
        sepsis = np.random.random() < 0.25
        risk = np.clip(sepsis * 0.7 + np.random.randn() * 0.2, 0, 1)
        
        outcome = CaseOutcome(
            stay_id=i,
            icu_unit="MICU",
            prediction_time=f"2150-01-{i+1:02d}",
            risk_score=risk,
            confidence=1.0 - abs(np.random.randn() * 0.2),
            alert_level="CRITICAL" if risk > 0.6 else ("AMBER" if risk > 0.3 else "WATCH"),
            red_team_triggered=np.random.random() < 0.1,
            n_tripwires=int(np.random.random() < 0.1) + int(np.random.random() < 0.1),
            actual_sepsis=sepsis,
        )
        
        feedback = agent.record_outcome(outcome)
        if feedback:
            for k, v in feedback.items():
                print(f"  [{k}] {v}")
    
    print(f"\n{agent.summary()}")
    
    weights = agent.get_near_miss_weights()
    if weights:
        print(f"\nNear-miss loss weights: {weights}")
    
    print("\n✓ Outcome Learning Agent test complete!")
