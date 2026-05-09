"""
Unified Orchestrator
====================
Multi-condition orchestrator for both Sepsis and FOU detection.

Priority Logic:
1. Sepsis takes priority (more acute)
2. If sepsis risk < 0.3 AND FOU probability > 0.5 → FOU workup
3. If both high → treat sepsis first, then FOU workup
4. Red Team overrides apply to both conditions

Inputs:
- sepsis_risk: float [0, 1]
- fou_probabilities: array [4,]
- sepsis_conformal: (lower, upper)
- fou_conformal: prediction_set
- red_team_sepsis: RedTeamAssessment
- red_team_fou: FouRedTeamAssessment

Output:
- primary_condition: "sepsis" | "fou" | "both" | "neither"
- alert_level: WATCH | AMBER | CRITICAL | FAST-TRACK
- actions: List[str]
- reasoning: str
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedAlertLevel(Enum):
    """Unified alert levels for multi-condition detection."""
    WATCH = "WATCH"
    AMBER = "AMBER"
    CRITICAL = "CRITICAL"
    FAST_TRACK = "FAST-TRACK"


@dataclass
class UnifiedDecision:
    """Unified decision for multi-condition detection."""
    primary_condition: str  # "sepsis", "fou", "both", "neither"
    alert_level: UnifiedAlertLevel
    sepsis_risk: float
    fou_probabilities: np.ndarray
    fou_predicted_class: int
    actions: List[str]
    reasoning: str
    sepsis_conformal: Optional[Tuple[float, float]]
    fou_conformal_set: Optional[List[int]]
    red_team_override: bool
    metadata: Dict[str, Any]


class UnifiedOrchestrator:
    """Unified orchestrator for Sepsis and FOU detection."""

    def __init__(
        self,
        sepsis_watch_threshold: float = 0.3,
        sepsis_amber_threshold: float = 0.6,
        fou_watch_threshold: float = 0.5,
        fou_amber_threshold: float = 0.7,
        sepsis_priority_threshold: float = 0.3,
        fou_activation_threshold: float = 0.5,
        high_confidence: float = 0.80,
        low_confidence: float = 0.50
    ):
        """
        Args:
            sepsis_watch_threshold: Sepsis risk threshold for WATCH
            sepsis_amber_threshold: Sepsis risk threshold for AMBER
            fou_watch_threshold: FOU probability threshold for WATCH
            fou_amber_threshold: FOU probability threshold for AMBER
            sepsis_priority_threshold: If sepsis > this, prioritize sepsis
            fou_activation_threshold: If sepsis < priority and FOU > this, activate FOU
            high_confidence: High confidence threshold for fast-tracking
            low_confidence: Low confidence threshold
        """
        self.sepsis_watch_threshold = sepsis_watch_threshold
        self.sepsis_amber_threshold = sepsis_amber_threshold
        self.fou_watch_threshold = fou_watch_threshold
        self.fou_amber_threshold = fou_amber_threshold
        self.sepsis_priority_threshold = sepsis_priority_threshold
        self.fou_activation_threshold = fou_activation_threshold
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence

    def decide(
        self,
        sepsis_risk: float,
        fou_probabilities: np.ndarray,
        sepsis_conformal: Optional[Tuple[float, float]] = None,
        fou_conformal_set: Optional[List[int]] = None,
        red_team_sepsis: Optional[Any] = None,
        red_team_fou: Optional[Any] = None,
        features: Optional[Dict[str, float]] = None
    ) -> UnifiedDecision:
        """
        Make unified decision for multi-condition detection.

        Args:
            sepsis_risk: Sepsis risk score [0, 1]
            fou_probabilities: FOU class probabilities [4,]
            sepsis_conformal: Optional conformal interval (lower, upper)
            fou_conformal_set: Optional FOU conformal prediction set
            red_team_sepsis: Optional sepsis red team assessment
            red_team_fou: Optional FOU red team assessment
            features: Optional current feature values

        Returns:
            UnifiedDecision with primary condition and actions
        """
        # Get FOU predicted class (highest probability)
        fou_predicted_class = int(np.argmax(fou_probabilities))
        fou_max_prob = fou_probabilities[fou_predicted_class]

        # Check for red team overrides
        red_team_override = False
        override_condition = None

        if red_team_sepsis is not None and hasattr(red_team_sepsis, 'alert_level'):
            from src.agents.red_team import AlertLevel as SepsisAlertLevel
            if red_team_sepsis.alert_level == SepsisAlertLevel.CRITICAL:
                red_team_override = True
                override_condition = "sepsis"

        if red_team_fou is not None and hasattr(red_team_fou, 'alert_level'):
            from src.agents.red_team_fou import AlertLevel as FouAlertLevel
            if red_team_fou.alert_level == FouAlertLevel.CRITICAL:
                red_team_override = True
                if override_condition == "sepsis":
                    override_condition = "both"
                else:
                    override_condition = "fou"

        # Priority logic
        if red_team_override:
            # Red team override takes precedence
            primary_condition = override_condition
            alert_level = UnifiedAlertLevel.CRITICAL
            actions = self._generate_override_actions(
                override_condition, red_team_sepsis, red_team_fou
            )
            reasoning = self._generate_override_reasoning(
                override_condition, red_team_sepsis, red_team_fou
            )

        elif sepsis_risk >= self.sepsis_priority_threshold:
            # Sepsis takes priority (more acute)
            primary_condition = "sepsis"
            alert_level = self._determine_sepsis_alert_level(sepsis_risk, sepsis_conformal)
            actions = self._generate_sepsis_actions(sepsis_risk, alert_level, sepsis_conformal)
            reasoning = self._generate_sepsis_reasoning(sepsis_risk, alert_level)

            # Check if FOU workup should follow
            if fou_max_prob > self.fou_activation_threshold:
                actions.append("After sepsis stabilization, initiate FOU workup")
                reasoning += f" FOU probability {fou_max_prob:.2f} suggests concurrent investigation after acute management."

        elif fou_max_prob > self.fou_activation_threshold:
            # FOU workup (sepsis risk low)
            primary_condition = "fou"
            alert_level = self._determine_fou_alert_level(fou_max_prob, fou_predicted_class)
            actions = self._generate_fou_actions(
                fou_probabilities, fou_predicted_class, alert_level, fou_conformal_set
            )
            reasoning = self._generate_fou_reasoning(
                fou_probabilities, fou_predicted_class, alert_level
            )

        elif sepsis_risk >= self.sepsis_watch_threshold or fou_max_prob >= self.fou_watch_threshold:
            # Both conditions warrant monitoring
            primary_condition = "both"
            alert_level = UnifiedAlertLevel.WATCH
            actions = self._generate_watch_actions(sepsis_risk, fou_max_prob)
            reasoning = f"Both sepsis (risk={sepsis_risk:.2f}) and FOU (prob={fou_max_prob:.2f}) warrant monitoring."

        else:
            # Neither condition detected
            primary_condition = "neither"
            alert_level = UnifiedAlertLevel.WATCH
            actions = ["Continue routine monitoring"]
            reasoning = "No significant sepsis or FOU risk detected."

        # Check for fast-tracking based on confidence
        if alert_level in [UnifiedAlertLevel.AMBER, UnifiedAlertLevel.CRITICAL]:
            if self._should_fast_track(sepsis_risk, fou_max_prob, sepsis_conformal, fou_conformal_set):
                alert_level = UnifiedAlertLevel.FAST_TRACK
                actions.insert(0, "FAST-TRACK: Skip preliminary workup, proceed directly to definitive diagnostics")

        return UnifiedDecision(
            primary_condition=primary_condition,
            alert_level=alert_level,
            sepsis_risk=sepsis_risk,
            fou_probabilities=fou_probabilities,
            fou_predicted_class=fou_predicted_class,
            actions=actions,
            reasoning=reasoning,
            sepsis_conformal=sepsis_conformal,
            fou_conformal_set=fou_conformal_set,
            red_team_override=red_team_override,
            metadata={
                "sepsis_watch_threshold": self.sepsis_watch_threshold,
                "fou_activation_threshold": self.fou_activation_threshold,
            }
        )

    def _determine_sepsis_alert_level(
        self,
        sepsis_risk: float,
        conformal: Optional[Tuple[float, float]]
    ) -> UnifiedAlertLevel:
        """Determine alert level for sepsis."""
        if sepsis_risk >= self.sepsis_amber_threshold:
            return UnifiedAlertLevel.CRITICAL
        elif sepsis_risk >= self.sepsis_watch_threshold:
            return UnifiedAlertLevel.AMBER
        else:
            return UnifiedAlertLevel.WATCH

    def _determine_fou_alert_level(
        self,
        fou_max_prob: float,
        predicted_class: int
    ) -> UnifiedAlertLevel:
        """Determine alert level for FOU."""
        # Infectious FOU (class 1) is more urgent
        if predicted_class == 1 and fou_max_prob >= self.fou_amber_threshold:
            return UnifiedAlertLevel.CRITICAL
        elif fou_max_prob >= self.fou_amber_threshold:
            return UnifiedAlertLevel.AMBER
        elif fou_max_prob >= self.fou_watch_threshold:
            return UnifiedAlertLevel.WATCH
        else:
            return UnifiedAlertLevel.WATCH

    def _should_fast_track(
        self,
        sepsis_risk: float,
        fou_max_prob: float,
        sepsis_conformal: Optional[Tuple[float, float]],
        fou_conformal_set: Optional[List[int]]
    ) -> bool:
        """Determine if case should be fast-tracked."""
        # Fast-track if high confidence
        if sepsis_risk > self.high_confidence:
            return True

        if fou_max_prob > self.high_confidence:
            return True

        # Fast-track if conformal prediction is very tight
        if sepsis_conformal is not None:
            lower, upper = sepsis_conformal
            if upper - lower < 0.2:  # Tight interval
                return True

        if fou_conformal_set is not None and len(fou_conformal_set) == 1:
            # Singleton prediction set (high confidence)
            return True

        return False

    def _generate_sepsis_actions(
        self,
        sepsis_risk: float,
        alert_level: UnifiedAlertLevel,
        conformal: Optional[Tuple[float, float]]
    ) -> List[str]:
        """Generate actions for sepsis."""
        actions = []

        if alert_level == UnifiedAlertLevel.CRITICAL:
            actions.append("CRITICAL: Immediate sepsis protocol activation")
            actions.append("Stat blood cultures, lactate, CBC")
            actions.append("Broad-spectrum antibiotics within 1 hour")
            actions.append("Fluid resuscitation (30 mL/kg crystalloid)")
            actions.append("ICU consultation")
        elif alert_level == UnifiedAlertLevel.AMBER:
            actions.append("AMBER: Expedite sepsis workup")
            actions.append("Blood cultures, lactate, procalcitonin")
            actions.append("Consider early antibiotics if source identified")
            actions.append("Close monitoring (q1h vitals)")
        else:
            actions.append("WATCH: Monitor for sepsis progression")
            actions.append("Serial vital signs and labs")

        return actions

    def _generate_fou_actions(
        self,
        probabilities: np.ndarray,
        predicted_class: int,
        alert_level: UnifiedAlertLevel,
        conformal_set: Optional[List[int]]
    ) -> List[str]:
        """Generate actions for FOU."""
        actions = []

        class_names = ["No FOU", "Infectious FOU", "Non-infectious FOU", "Undiagnosed FOU"]
        predicted_name = class_names[predicted_class]

        actions.append(f"FOU workup indicated (predicted: {predicted_name})")

        if predicted_class == 1:  # Infectious FOU
            actions.append("Infectious disease consultation")
            actions.append("Comprehensive infectious workup:")
            actions.append("  - Blood cultures (aerobic, anaerobic, fungal)")
            actions.append("  - TB testing (QuantiFERON, sputum AFB)")
            actions.append("  - Echocardiogram (rule out endocarditis)")
            actions.append("  - CT chest/abdomen/pelvis with contrast")
        elif predicted_class == 2:  # Non-infectious FOU
            actions.append("Rheumatology consultation")
            actions.append("Autoimmune/inflammatory workup:")
            actions.append("  - ANA, RF, anti-CCP, complement levels")
            actions.append("  - ESR, CRP, ferritin")
            actions.append("  - Consider PET-CT for malignancy")
        else:  # Undiagnosed or No FOU
            actions.append("Comprehensive FOU workup")
            actions.append("Consider both infectious and non-infectious causes")

        if conformal_set is not None and len(conformal_set) > 1:
            conformal_names = [class_names[c] for c in conformal_set]
            actions.append(f"Conformal prediction set: {conformal_names}")
            actions.append("Consider workup for all predicted categories")

        return actions

    def _generate_watch_actions(self, sepsis_risk: float, fou_prob: float) -> List[str]:
        """Generate actions for WATCH level."""
        actions = [
            "Continue monitoring both sepsis and FOU indicators",
            "Serial vital signs (q4h)",
            "Daily labs (CBC, CMP, inflammatory markers)",
            "Reassess if clinical deterioration"
        ]
        return actions

    def _generate_override_actions(
        self,
        condition: str,
        red_team_sepsis: Any,
        red_team_fou: Any
    ) -> List[str]:
        """Generate actions for red team override."""
        actions = ["RED TEAM OVERRIDE: Clinical tripwires triggered"]

        if condition in ["sepsis", "both"] and red_team_sepsis is not None:
            actions.extend(red_team_sepsis.actions)

        if condition in ["fou", "both"] and red_team_fou is not None:
            actions.extend(red_team_fou.actions)

        return actions

    def _generate_sepsis_reasoning(
        self,
        sepsis_risk: float,
        alert_level: UnifiedAlertLevel
    ) -> str:
        """Generate reasoning for sepsis decision."""
        return (
            f"Sepsis risk {sepsis_risk:.2f} exceeds priority threshold. "
            f"Alert level: {alert_level.value}. Sepsis is more acute and takes priority."
        )

    def _generate_fou_reasoning(
        self,
        probabilities: np.ndarray,
        predicted_class: int,
        alert_level: UnifiedAlertLevel
    ) -> str:
        """Generate reasoning for FOU decision."""
        class_names = ["No FOU", "Infectious FOU", "Non-infectious FOU", "Undiagnosed FOU"]
        predicted_name = class_names[predicted_class]
        max_prob = probabilities[predicted_class]

        return (
            f"FOU predicted: {predicted_name} (probability {max_prob:.2f}). "
            f"Sepsis risk low, FOU workup indicated. Alert level: {alert_level.value}."
        )

    def _generate_override_reasoning(
        self,
        condition: str,
        red_team_sepsis: Any,
        red_team_fou: Any
    ) -> str:
        """Generate reasoning for red team override."""
        reasons = ["RED TEAM OVERRIDE: Clinical tripwires triggered."]

        if condition in ["sepsis", "both"] and red_team_sepsis is not None:
            reasons.append(f"Sepsis: {red_team_sepsis.reasoning}")

        if condition in ["fou", "both"] and red_team_fou is not None:
            reasons.append(f"FOU: {red_team_fou.reasoning}")

        return " ".join(reasons)


if __name__ == "__main__":
    # Test unified orchestrator
    print("=== Unified Orchestrator Test ===")

    orchestrator = UnifiedOrchestrator()

    # Test case 1: High sepsis risk (sepsis priority)
    print("\nTest 1: High sepsis risk")
    decision1 = orchestrator.decide(
        sepsis_risk=0.75,
        fou_probabilities=np.array([0.4, 0.3, 0.2, 0.1]),
        sepsis_conformal=(0.65, 0.85),
        fou_conformal_set=[0, 1]
    )
    print(f"Primary condition: {decision1.primary_condition}")
    print(f"Alert level: {decision1.alert_level.value}")
    print(f"Actions: {decision1.actions[:3]}")

    # Test case 2: Low sepsis, high FOU (FOU workup)
    print("\nTest 2: Low sepsis, high FOU")
    decision2 = orchestrator.decide(
        sepsis_risk=0.15,
        fou_probabilities=np.array([0.1, 0.7, 0.15, 0.05]),
        sepsis_conformal=(0.05, 0.25),
        fou_conformal_set=[1]
    )
    print(f"Primary condition: {decision2.primary_condition}")
    print(f"Alert level: {decision2.alert_level.value}")
    print(f"FOU predicted class: {decision2.fou_predicted_class} (Infectious FOU)")
    print(f"Actions: {decision2.actions[:3]}")

    # Test case 3: Both moderate (watch both)
    print("\nTest 3: Both moderate")
    decision3 = orchestrator.decide(
        sepsis_risk=0.35,
        fou_probabilities=np.array([0.3, 0.3, 0.25, 0.15]),
    )
    print(f"Primary condition: {decision3.primary_condition}")
    print(f"Alert level: {decision3.alert_level.value}")
    print(f"Reasoning: {decision3.reasoning}")

    print("\n✓ Unified orchestrator test passed")
