"""
FOU Red Team Agent
==================
Safety agent with FOU-specific clinical tripwires.

5 Clinical Tripwires:
1. TW-FEVER: Persistent fever (>39°C for >72 hours)
2. TW-SEPSIS: Sepsis risk (qSOFA ≥ 2) - rule out sepsis first
3. TW-NEUTROPENIA: Neutropenia (ANC < 500) - high infection risk
4. TW-HYPOTENSION: Hypotension (MAP < 65 mmHg)
5. TW-ALTERED: Altered mental status (GCS < 13)

Escalation Logic:
- ≥ 2 tripwires → CRITICAL (immediate infectious disease consult)
- 1 tripwire → AMBER (expedite workup)
- 0 tripwires → WATCH (routine monitoring)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for FOU detection."""
    WATCH = "WATCH"
    AMBER = "AMBER"
    CRITICAL = "CRITICAL"


@dataclass
class TripwireAssessment:
    """Assessment of a single tripwire."""
    name: str
    triggered: bool
    value: float
    threshold: float
    reason: str


@dataclass
class FouRedTeamAssessment:
    """Complete red team assessment for FOU."""
    alert_level: AlertLevel
    tripwires_triggered: List[TripwireAssessment]
    n_tripwires: int
    actions: List[str]
    reasoning: str


class FouRedTeamAgent:
    """Red Team Agent for FOU detection with clinical tripwires."""

    def __init__(
        self,
        persistent_fever_temp: float = 39.0,
        persistent_fever_hours: float = 72.0,
        qsofa_threshold: int = 2,
        anc_threshold: float = 500.0,
        map_threshold: float = 65.0,
        gcs_threshold: float = 13.0,
        critical_tripwire_count: int = 2
    ):
        """
        Args:
            persistent_fever_temp: Temperature threshold for persistent fever (°C)
            persistent_fever_hours: Duration threshold for persistent fever (hours)
            qsofa_threshold: qSOFA score threshold for sepsis rule-out
            anc_threshold: Absolute neutrophil count threshold
            map_threshold: Mean arterial pressure threshold (mmHg)
            gcs_threshold: Glasgow Coma Scale threshold
            critical_tripwire_count: Number of tripwires for CRITICAL alert
        """
        self.persistent_fever_temp = persistent_fever_temp
        self.persistent_fever_hours = persistent_fever_hours
        self.qsofa_threshold = qsofa_threshold
        self.anc_threshold = anc_threshold
        self.map_threshold = map_threshold
        self.gcs_threshold = gcs_threshold
        self.critical_tripwire_count = critical_tripwire_count

    def assess(
        self,
        features: Dict[str, float],
        window_features: Optional[np.ndarray] = None
    ) -> FouRedTeamAssessment:
        """
        Assess FOU patient for clinical tripwires.

        Args:
            features: Current feature values (dict with feature names)
            window_features: Optional full window for temporal analysis, shape (seq_len, n_features)

        Returns:
            FouRedTeamAssessment with alert level and actions
        """
        tripwires = []

        # Tripwire 1: Persistent Fever
        tw_fever = self._check_persistent_fever(features, window_features)
        tripwires.append(tw_fever)

        # Tripwire 2: Sepsis Risk (qSOFA)
        tw_sepsis = self._check_sepsis_risk(features)
        tripwires.append(tw_sepsis)

        # Tripwire 3: Neutropenia
        tw_neutropenia = self._check_neutropenia(features)
        tripwires.append(tw_neutropenia)

        # Tripwire 4: Hypotension
        tw_hypotension = self._check_hypotension(features)
        tripwires.append(tw_hypotension)

        # Tripwire 5: Altered Mental Status
        tw_altered = self._check_altered_mental_status(features)
        tripwires.append(tw_altered)

        # Count triggered tripwires
        triggered = [tw for tw in tripwires if tw.triggered]
        n_tripwires = len(triggered)

        # Determine alert level
        if n_tripwires >= self.critical_tripwire_count:
            alert_level = AlertLevel.CRITICAL
        elif n_tripwires == 1:
            alert_level = AlertLevel.AMBER
        else:
            alert_level = AlertLevel.WATCH

        # Generate actions
        actions = self._generate_actions(alert_level, triggered)

        # Generate reasoning
        reasoning = self._generate_reasoning(alert_level, triggered)

        return FouRedTeamAssessment(
            alert_level=alert_level,
            tripwires_triggered=triggered,
            n_tripwires=n_tripwires,
            actions=actions,
            reasoning=reasoning
        )

    def _check_persistent_fever(
        self,
        features: Dict[str, float],
        window_features: Optional[np.ndarray]
    ) -> TripwireAssessment:
        """Check for persistent fever (>39°C for >72 hours)."""
        # Check current temperature
        temp = features.get('temperature', 0.0)

        # Check fever duration if available
        fever_duration = features.get('fever_duration_hours', 0.0)

        triggered = (temp > self.persistent_fever_temp and
                    fever_duration > self.persistent_fever_hours)

        reason = f"Temperature {temp:.1f}°C for {fever_duration:.0f} hours"

        return TripwireAssessment(
            name="TW-FEVER",
            triggered=triggered,
            value=temp,
            threshold=self.persistent_fever_temp,
            reason=reason
        )

    def _check_sepsis_risk(self, features: Dict[str, float]) -> TripwireAssessment:
        """Check for sepsis risk using qSOFA criteria."""
        # qSOFA criteria:
        # 1. Respiratory rate ≥ 22
        # 2. Altered mentation (GCS < 15)
        # 3. Systolic BP ≤ 100

        qsofa_score = 0

        rr = features.get('resp_rate', 0.0)
        if rr >= 22.0:
            qsofa_score += 1

        gcs = features.get('gcs_total', 15.0)
        if gcs < 15.0:
            qsofa_score += 1

        sbp = features.get('sbp', 120.0)
        if sbp <= 100.0:
            qsofa_score += 1

        triggered = qsofa_score >= self.qsofa_threshold

        reason = f"qSOFA score {qsofa_score} (RR={rr:.0f}, GCS={gcs:.0f}, SBP={sbp:.0f})"

        return TripwireAssessment(
            name="TW-SEPSIS",
            triggered=triggered,
            value=float(qsofa_score),
            threshold=float(self.qsofa_threshold),
            reason=reason
        )

    def _check_neutropenia(self, features: Dict[str, float]) -> TripwireAssessment:
        """Check for neutropenia (ANC < 500)."""
        # ANC = WBC × (% neutrophils)
        # Simplified: use WBC as proxy (normal WBC ~4-11 K/μL)
        wbc = features.get('wbc', 5.0)

        # Estimate ANC (assuming ~60% neutrophils normally)
        anc_estimate = wbc * 0.6 * 1000  # Convert to cells/μL

        triggered = anc_estimate < self.anc_threshold

        reason = f"Estimated ANC {anc_estimate:.0f} cells/μL (WBC={wbc:.1f} K/μL)"

        return TripwireAssessment(
            name="TW-NEUTROPENIA",
            triggered=triggered,
            value=anc_estimate,
            threshold=self.anc_threshold,
            reason=reason
        )

    def _check_hypotension(self, features: Dict[str, float]) -> TripwireAssessment:
        """Check for hypotension (MAP < 65 mmHg)."""
        map_value = features.get('map', 80.0)

        triggered = map_value < self.map_threshold

        reason = f"MAP {map_value:.0f} mmHg"

        return TripwireAssessment(
            name="TW-HYPOTENSION",
            triggered=triggered,
            value=map_value,
            threshold=self.map_threshold,
            reason=reason
        )

    def _check_altered_mental_status(self, features: Dict[str, float]) -> TripwireAssessment:
        """Check for altered mental status (GCS < 13)."""
        gcs = features.get('gcs_total', 15.0)

        triggered = gcs < self.gcs_threshold

        reason = f"GCS {gcs:.0f}"

        return TripwireAssessment(
            name="TW-ALTERED",
            triggered=triggered,
            value=gcs,
            threshold=self.gcs_threshold,
            reason=reason
        )

    def _generate_actions(
        self,
        alert_level: AlertLevel,
        triggered: List[TripwireAssessment]
    ) -> List[str]:
        """Generate recommended actions based on alert level."""
        actions = []

        if alert_level == AlertLevel.CRITICAL:
            actions.append("IMMEDIATE infectious disease consult")
            actions.append("Stat blood cultures (if not already done)")
            actions.append("Consider empiric broad-spectrum antibiotics")
            actions.append("Expedite imaging (CT chest/abdomen/pelvis)")
            actions.append("Review medication list for drug fever")

            # Specific actions based on tripwires
            for tw in triggered:
                if tw.name == "TW-SEPSIS":
                    actions.append("Rule out sepsis FIRST - consider ICU transfer")
                elif tw.name == "TW-NEUTROPENIA":
                    actions.append("Neutropenic fever protocol - isolation precautions")
                elif tw.name == "TW-HYPOTENSION":
                    actions.append("Fluid resuscitation - consider vasopressors")

        elif alert_level == AlertLevel.AMBER:
            actions.append("Expedite FOU workup")
            actions.append("Daily blood cultures")
            actions.append("Consider advanced imaging (PET-CT if available)")
            actions.append("Rheumatology consult if inflammatory markers elevated")

            for tw in triggered:
                if tw.name == "TW-FEVER":
                    actions.append("Consider antipyretics for comfort")
                elif tw.name == "TW-ALTERED":
                    actions.append("Neurology consult - consider LP if CNS infection suspected")

        else:  # WATCH
            actions.append("Continue routine FOU monitoring")
            actions.append("Serial blood cultures every 48 hours")
            actions.append("Monitor temperature curve and inflammatory markers")

        return actions

    def _generate_reasoning(
        self,
        alert_level: AlertLevel,
        triggered: List[TripwireAssessment]
    ) -> str:
        """Generate reasoning for the assessment."""
        if not triggered:
            return "No clinical tripwires triggered. Continue routine FOU workup."

        tripwire_names = [tw.name for tw in triggered]
        tripwire_str = ", ".join(tripwire_names)

        if alert_level == AlertLevel.CRITICAL:
            return (
                f"CRITICAL: {len(triggered)} tripwires triggered ({tripwire_str}). "
                f"Patient requires immediate escalation and infectious disease consultation. "
                f"High risk of serious underlying infection or inflammatory condition."
            )
        elif alert_level == AlertLevel.AMBER:
            return (
                f"AMBER: 1 tripwire triggered ({tripwire_str}). "
                f"Expedite FOU workup to identify underlying cause. "
                f"Monitor closely for additional concerning features."
            )
        else:
            return "WATCH: Continue monitoring."


if __name__ == "__main__":
    # Test FOU Red Team Agent
    print("=== FOU Red Team Agent Test ===")

    agent = FouRedTeamAgent()

    # Test case 1: No tripwires
    print("\nTest 1: No tripwires")
    features1 = {
        'temperature': 38.5,
        'fever_duration_hours': 48.0,
        'resp_rate': 18.0,
        'gcs_total': 15.0,
        'sbp': 120.0,
        'map': 80.0,
        'wbc': 8.0,
    }
    assessment1 = agent.assess(features1)
    print(f"Alert level: {assessment1.alert_level.value}")
    print(f"Tripwires: {assessment1.n_tripwires}")
    print(f"Actions: {assessment1.actions}")

    # Test case 2: Persistent fever + neutropenia (CRITICAL)
    print("\nTest 2: Persistent fever + neutropenia (CRITICAL)")
    features2 = {
        'temperature': 39.5,
        'fever_duration_hours': 96.0,
        'resp_rate': 18.0,
        'gcs_total': 15.0,
        'sbp': 120.0,
        'map': 80.0,
        'wbc': 0.5,  # Severe neutropenia
    }
    assessment2 = agent.assess(features2)
    print(f"Alert level: {assessment2.alert_level.value}")
    print(f"Tripwires: {assessment2.n_tripwires}")
    print(f"Triggered: {[tw.name for tw in assessment2.tripwires_triggered]}")
    print(f"Reasoning: {assessment2.reasoning}")
    print(f"Actions: {assessment2.actions[:3]}")

    # Test case 3: High qSOFA (sepsis risk)
    print("\nTest 3: High qSOFA (sepsis risk)")
    features3 = {
        'temperature': 38.8,
        'fever_duration_hours': 60.0,
        'resp_rate': 24.0,  # ≥22
        'gcs_total': 13.0,  # <15
        'sbp': 95.0,  # ≤100
        'map': 70.0,
        'wbc': 12.0,
    }
    assessment3 = agent.assess(features3)
    print(f"Alert level: {assessment3.alert_level.value}")
    print(f"Tripwires: {assessment3.n_tripwires}")
    print(f"Triggered: {[tw.name for tw in assessment3.tripwires_triggered]}")

    print("\n✓ FOU Red Team Agent test passed")
