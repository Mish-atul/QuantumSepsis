"""
QuantumSepsis Shield — Confidence-Gated Intervention Orchestrator
==================================================================

Combines quantum risk score, conformal prediction intervals,
and Red Team Agent assessments into clinical action decisions.

Novelty 3: Confidence-Gated Diagnostic Fast-Tracking
  - confidence > 0.80 AND risk > 0.6 → SKIP preliminary diagnostics,
    immediately trigger PCT + blood culture + vasopressor protocol
  - confidence 0.5–0.80 → standard AMBER parallel ordering
  - confidence < 0.5 → flag for manual clinician review
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import OrchestratorConfig, get_default_config
from src.agents.red_team import RedTeamAssessment

logger = logging.getLogger(__name__)


@dataclass
class ClinicalAction:
    """A clinical action to be triggered."""
    action_type: str        # "ORDER", "NOTIFY", "PAGE", "MONITOR"
    description: str
    priority: int           # 1 (highest) to 5 (lowest)
    estimated_cost: str     # "LOW", "MEDIUM", "HIGH"
    time_sensitivity: str   # "IMMEDIATE", "URGENT", "ROUTINE"


@dataclass
class OrchestratorDecision:
    """Complete orchestrator output for a patient assessment cycle."""
    alert_level: str        # "WATCH", "AMBER", "CRITICAL"
    risk_score: float
    confidence: float
    conformal_lower: float
    conformal_upper: float
    conformal_width: float
    red_team_override: str
    fast_tracked: bool
    actions: List[ClinicalAction]
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "alert_level": self.alert_level,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "conformal_interval": [
                round(self.conformal_lower, 4),
                round(self.conformal_upper, 4),
            ],
            "conformal_width": round(self.conformal_width, 4),
            "red_team_override": self.red_team_override,
            "fast_tracked": self.fast_tracked,
            "actions": [
                {"type": a.action_type, "description": a.description, "priority": a.priority}
                for a in self.actions
            ],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


# Standard clinical action sets
WATCH_ACTIONS = [
    ClinicalAction("MONITOR", "Dashboard refresh — reassess in 15 min", 5, "LOW", "ROUTINE"),
]

AMBER_ACTIONS = [
    ClinicalAction("ORDER", "Stat lactate level", 2, "LOW", "URGENT"),
    ClinicalAction("ORDER", "Procalcitonin (PCT) level", 2, "MEDIUM", "URGENT"),
    ClinicalAction("ORDER", "Blood culture × 2 sets", 2, "MEDIUM", "URGENT"),
    ClinicalAction("NOTIFY", "Notify attending physician", 2, "LOW", "URGENT"),
    ClinicalAction("MONITOR", "Increase monitoring frequency to q5min", 3, "LOW", "URGENT"),
]

CRITICAL_ACTIONS = [
    ClinicalAction("PAGE", "IMMEDIATE attending/fellow page — suspected sepsis", 1, "LOW", "IMMEDIATE"),
    ClinicalAction("ORDER", "Blood culture × 2 sets (before antibiotics)", 1, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "Stat lactate level", 1, "LOW", "IMMEDIATE"),
    ClinicalAction("ORDER", "Broad-spectrum antibiotics within 1 hour", 1, "HIGH", "IMMEDIATE"),
    ClinicalAction("ORDER", "30 mL/kg crystalloid bolus if MAP < 65", 1, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "Procalcitonin (PCT) level", 2, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "CBC, CMP, coagulation panel", 2, "MEDIUM", "URGENT"),
    ClinicalAction("MONITOR", "Continuous monitoring — q5min vital signs", 1, "LOW", "IMMEDIATE"),
]

FAST_TRACK_ACTIONS = [
    ClinicalAction("PAGE", "IMMEDIATE attending/fellow page — HIGH CONFIDENCE sepsis alert", 1, "LOW", "IMMEDIATE"),
    ClinicalAction("ORDER", "Blood culture × 2 sets (before antibiotics)", 1, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "SKIP CBC wait — immediate procalcitonin (PCT)", 1, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "Broad-spectrum antibiotics within 1 hour", 1, "HIGH", "IMMEDIATE"),
    ClinicalAction("ORDER", "30 mL/kg crystalloid bolus", 1, "MEDIUM", "IMMEDIATE"),
    ClinicalAction("ORDER", "Vasopressor preparation (norepinephrine standby)", 1, "HIGH", "IMMEDIATE"),
    ClinicalAction("MONITOR", "Continuous monitoring — arterial line placement", 1, "MEDIUM", "IMMEDIATE"),
]


class ConfidenceGatedOrchestrator:
    """Confidence-gated clinical intervention orchestrator.
    
    Combines:
        1. Quantum risk score (or classical LSTM score during Phase 1)
        2. Conformal prediction intervals
        3. Red Team Agent assessment
    
    Into a final clinical decision with appropriate action set.
    
    The key innovation (Novelty 3) is using conformal interval width
    as a calibrated confidence proxy to skip low-value diagnostic steps.
    
    Args:
        config: OrchestratorConfig with threshold values
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        if config is None:
            config = get_default_config().orchestrator
        self.config = config
    
    def decide(
        self,
        risk_score: float,
        conformal_lower: float,
        conformal_upper: float,
        red_team: RedTeamAssessment,
    ) -> OrchestratorDecision:
        """Make a clinical decision combining all system inputs.
        
        Args:
            risk_score: Point risk estimate [0, 1]
            conformal_lower: Lower bound of conformal interval
            conformal_upper: Upper bound of conformal interval
            red_team: Red Team Agent assessment
        
        Returns:
            OrchestratorDecision with alert level, actions, and reasoning
        """
        # Compute confidence from conformal width
        conformal_width = conformal_upper - conformal_lower
        confidence = max(0.0, min(1.0, 1.0 - conformal_width))
        
        # Initialize reasoning log
        reasoning_parts = []
        
        # --- Step 1: Red Team Override (NON-OVERRIDABLE) ---
        if red_team.override_level == "CRITICAL":
            reasoning_parts.append(
                f"RED TEAM OVERRIDE: {red_team.n_active} tripwires active "
                f"— forcing CRITICAL regardless of model output"
            )
            return OrchestratorDecision(
                alert_level="CRITICAL",
                risk_score=risk_score,
                confidence=confidence,
                conformal_lower=conformal_lower,
                conformal_upper=conformal_upper,
                conformal_width=conformal_width,
                red_team_override="CRITICAL",
                fast_tracked=False,
                actions=CRITICAL_ACTIONS,
                reasoning=" | ".join(reasoning_parts),
            )
        
        # --- Step 2: Confidence-Gated Fast-Tracking (Novelty 3) ---
        fast_tracked = False
        
        if (confidence > self.config.high_confidence and
                risk_score > self.config.amber_threshold):
            # HIGH CONFIDENCE + HIGH RISK → Skip preliminary diagnostics
            fast_tracked = True
            reasoning_parts.append(
                f"FAST-TRACK: Confidence={confidence:.2f} > {self.config.high_confidence}, "
                f"Risk={risk_score:.2f} > {self.config.amber_threshold} "
                f"— skipping CBC wait, immediate PCT + culture + vasopressor protocol"
            )
            alert_level = "CRITICAL"
            actions = FAST_TRACK_ACTIONS
        
        elif confidence < self.config.low_confidence:
            # LOW CONFIDENCE → Manual clinician review
            reasoning_parts.append(
                f"LOW CONFIDENCE: {confidence:.2f} < {self.config.low_confidence} "
                f"— flagging for manual clinician review with uncertainty report"
            )
            alert_level = "AMBER"
            actions = AMBER_ACTIONS + [
                ClinicalAction(
                    "NOTIFY",
                    f"UNCERTAINTY ALERT: Model confidence low ({confidence:.0%}). "
                    f"Conformal interval: [{conformal_lower:.2f}, {conformal_upper:.2f}]. "
                    f"Manual assessment recommended.",
                    2, "LOW", "URGENT"
                ),
            ]
        
        elif risk_score > self.config.amber_threshold:
            # HIGH RISK, MODERATE CONFIDENCE → CRITICAL
            reasoning_parts.append(
                f"HIGH RISK: {risk_score:.2f} > {self.config.amber_threshold}"
            )
            alert_level = "CRITICAL"
            actions = CRITICAL_ACTIONS
        
        elif risk_score > self.config.watch_threshold:
            # MODERATE RISK → AMBER
            reasoning_parts.append(
                f"MODERATE RISK: {risk_score:.2f} in [{self.config.watch_threshold}, {self.config.amber_threshold}]"
            )
            alert_level = "AMBER"
            actions = AMBER_ACTIONS
        
        else:
            # LOW RISK → WATCH
            reasoning_parts.append(
                f"LOW RISK: {risk_score:.2f} < {self.config.watch_threshold}"
            )
            alert_level = "WATCH"
            actions = WATCH_ACTIONS
        
        # Escalate if Red Team says AMBER
        if red_team.override_level == "AMBER" and alert_level == "WATCH":
            reasoning_parts.append(
                f"Red Team AMBER escalation: {red_team.n_active} tripwire(s) active"
            )
            alert_level = "AMBER"
            actions = AMBER_ACTIONS
        
        # Escalate if conformal width indicates high uncertainty
        if conformal_width > 0.4 and alert_level == "WATCH":
            reasoning_parts.append(
                f"UNCERTAINTY ESCALATION: Conformal width={conformal_width:.2f} > 0.4"
            )
            alert_level = "AMBER"
            actions = AMBER_ACTIONS
        
        return OrchestratorDecision(
            alert_level=alert_level,
            risk_score=risk_score,
            confidence=confidence,
            conformal_lower=conformal_lower,
            conformal_upper=conformal_upper,
            conformal_width=conformal_width,
            red_team_override=red_team.override_level,
            fast_tracked=fast_tracked,
            actions=actions,
            reasoning=" | ".join(reasoning_parts),
        )


def test_orchestrator():
    """Test the orchestrator with various scenarios."""
    from src.agents.red_team import RedTeamAgent, TripwireResult
    import numpy as np
    
    orchestrator = ConfidenceGatedOrchestrator()
    
    # Helper to create a simple RedTeamAssessment
    def make_rt(override="WATCH", n_active=0):
        return RedTeamAssessment(
            triggered=n_active > 0,
            active_tripwires=[],
            override_level=override,
            n_active=n_active,
            details="",
        )
    
    print("Orchestrator — Test Scenarios")
    print("=" * 60)
    
    # 1. Low risk, no tripwires
    d = orchestrator.decide(0.1, 0.05, 0.20, make_rt())
    print(f"\n1. Low risk (0.1): {d.alert_level}")
    assert d.alert_level == "WATCH"
    
    # 2. Moderate risk, moderate confidence
    d = orchestrator.decide(0.45, 0.30, 0.60, make_rt())
    print(f"2. Moderate risk (0.45): {d.alert_level}")
    assert d.alert_level == "AMBER"
    
    # 3. High risk, high confidence → FAST TRACK
    d = orchestrator.decide(0.85, 0.80, 0.90, make_rt())
    print(f"3. High risk + high confidence: {d.alert_level}, fast_tracked={d.fast_tracked}")
    assert d.alert_level == "CRITICAL"
    assert d.fast_tracked == True
    
    # 4. Any risk, 2+ tripwires → CRITICAL override
    d = orchestrator.decide(0.1, 0.05, 0.20, make_rt("CRITICAL", 2))
    print(f"4. Low risk + 2 tripwires: {d.alert_level}")
    assert d.alert_level == "CRITICAL"
    
    # 5. Low risk, wide conformal interval → uncertainty escalation
    d = orchestrator.decide(0.15, 0.0, 0.60, make_rt())
    print(f"5. Low risk + wide interval: {d.alert_level}")
    assert d.alert_level == "AMBER"
    
    # 6. Low confidence → clinician review
    d = orchestrator.decide(0.5, 0.1, 0.9, make_rt())
    print(f"6. Moderate risk + low confidence: {d.alert_level}")
    assert d.alert_level == "AMBER"
    
    print(f"\n✓ All orchestrator tests passed!")
    
    # Print a detailed decision
    print(f"\nDetailed decision (Scenario 3 — FAST TRACK):")
    d = orchestrator.decide(0.85, 0.80, 0.90, make_rt())
    for k, v in d.to_dict().items():
        if k == "actions":
            print(f"  {k}:")
            for a in v:
                print(f"    - [{a['priority']}] {a['type']}: {a['description']}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test_orchestrator()
