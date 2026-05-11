"""
QuantumSepsis Shield — Red Team Agent (Adversarial Safety Layer)
================================================================

Independent, non-overridable safety agent with clinical tripwires.
CANNOT be suppressed by ML model output.

Tripwires:
  TW-TEMP:   Temperature < 36°C or > 38.3°C
  TW-HR:     Heart Rate > 100 bpm (or > 130 with no trend required)
  TW-RR:     Respiratory Rate > 22 breaths/min
  TW-MAP:    MAP < 65 mmHg
  TW-SPO2:   SpO2 < 92%
  TW-LACTATE: Lactate > 2.0 mmol/L
  TW-MENTAL: GCS < 13 (altered mental status)

Escalation:
  Any SINGLE extreme vital → CRITICAL (non-overridable)
  ≥ 2 tripwires active → CRITICAL (non-overridable)
  1 tripwire → AMBER
  0 tripwires → WATCH
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import RedTeamConfig, get_default_config

logger = logging.getLogger(__name__)


# Feature indices in the 12-feature vector
# Must match the order in config.data.feature_names
FEAT_IDX = {
    "heart_rate": 0,
    "sbp": 1,
    "dbp": 2,
    "map": 3,
    "temperature": 4,
    "resp_rate": 5,
    "spo2": 6,
    "gcs_total": 7,
    "lactate": 8,
    "wbc": 9,
    "creatinine": 10,
    "platelets": 11,
}


@dataclass
class TripwireResult:
    """Result of a single tripwire evaluation."""
    name: str
    triggered: bool
    value: float
    threshold: str
    clinical_reason: str


@dataclass
class RedTeamAssessment:
    """Complete Red Team Agent assessment output."""
    triggered: bool
    active_tripwires: List[TripwireResult]
    override_level: str  # "WATCH", "AMBER", "CRITICAL"
    n_active: int
    details: str
    
    def to_dict(self) -> Dict:
        return {
            "triggered": self.triggered,
            "override_level": self.override_level,
            "n_active_tripwires": self.n_active,
            "active_tripwires": [
                {"name": tw.name, "value": tw.value, "threshold": tw.threshold}
                for tw in self.active_tripwires if tw.triggered
            ],
            "details": self.details,
        }


class RedTeamAgent:
    """Independent adversarial safety agent with clinical tripwires.
    
    This agent operates COMPLETELY INDEPENDENTLY of the ML pipeline.
    Its decisions CANNOT be overridden by model output.
    
    Design principle: Even if the quantum kernel says risk = 0.0,
    two simultaneous tripwires force a CRITICAL alert.
    
    Args:
        config: RedTeamConfig with threshold values
        use_normalized: If True, input is z-normalized; apply denormalization
        norm_stats: Dict with train_mean and train_std for denormalization
    """
    
    def __init__(
        self,
        config: Optional[RedTeamConfig] = None,
        use_normalized: bool = False,
        norm_stats: Optional[Dict] = None,
    ):
        if config is None:
            config = get_default_config().red_team
        
        self.config = config
        self.use_normalized = use_normalized
        self.norm_stats = norm_stats
    
    def evaluate(
        self,
        vitals_window: np.ndarray,
    ) -> RedTeamAssessment:
        """Evaluate clinical tripwires on a vitals window.
        
        Args:
            vitals_window: (T, 12) array — T time steps of 12 features.
                          Uses raw values if use_normalized=False,
                          otherwise denormalizes first.
        
        Returns:
            RedTeamAssessment with triggered status, active tripwires,
            and override level.
        """
        assert vitals_window.ndim == 2, f"Expected 2D array, got {vitals_window.ndim}D"
        assert vitals_window.shape[1] == 12, f"Expected 12 features, got {vitals_window.shape[1]}"
        
        # Denormalize if needed
        if self.use_normalized and self.norm_stats:
            vitals_window = self._denormalize(vitals_window)
        
        # Get latest readings and compute trends
        latest = vitals_window[-1]
        
        tripwires = []
        
        # TW-TEMP: Temperature
        temp = latest[FEAT_IDX["temperature"]]
        tripwires.append(TripwireResult(
            name="TW-TEMP",
            triggered=temp < self.config.temp_low or temp > self.config.temp_high,
            value=temp,
            threshold=f"< {self.config.temp_low}°C or > {self.config.temp_high}°C",
            clinical_reason="Hypothermia or fever (SIRS criterion)",
        ))
        
        # TW-HR: Heart Rate — fires on high value alone OR value + trend
        hr = latest[FEAT_IDX["heart_rate"]]
        hr_trend = self._compute_trend(vitals_window[:, FEAT_IDX["heart_rate"]])
        hr_extreme = hr > 130  # Extreme tachycardia — always critical regardless of trend
        hr_with_trend = (hr > self.config.hr_threshold and hr_trend > self.config.hr_trend_threshold)
        tripwires.append(TripwireResult(
            name="TW-HR",
            triggered=hr_extreme or hr_with_trend,
            value=hr,
            threshold=f"> 130 bpm (extreme) or > {self.config.hr_threshold} bpm + trend",
            clinical_reason="Tachycardia" + (" (EXTREME)" if hr_extreme else " with upward trend"),
        ))
        
        # TW-RR: Respiratory Rate
        rr = latest[FEAT_IDX["resp_rate"]]
        tripwires.append(TripwireResult(
            name="TW-RR",
            triggered=rr > self.config.rr_threshold,
            value=rr,
            threshold=f"> {self.config.rr_threshold} breaths/min",
            clinical_reason="Tachypnea (SIRS criterion)",
        ))
        
        # TW-MAP: Mean Arterial Pressure
        map_val = latest[FEAT_IDX["map"]]
        tripwires.append(TripwireResult(
            name="TW-MAP",
            triggered=map_val < self.config.map_threshold,
            value=map_val,
            threshold=f"< {self.config.map_threshold} mmHg",
            clinical_reason="Hypotension (Sepsis-3 cardiovascular)",
        ))
        
        # TW-SPO2: Oxygen saturation
        spo2 = latest[FEAT_IDX["spo2"]]
        tripwires.append(TripwireResult(
            name="TW-SPO2",
            triggered=spo2 < 92.0,
            value=spo2,
            threshold="< 92%",
            clinical_reason="Hypoxemia (respiratory failure risk)",
        ))
        
        # TW-LACTATE: Lactate level
        lactate = latest[FEAT_IDX["lactate"]]
        tripwires.append(TripwireResult(
            name="TW-LACTATE",
            triggered=lactate > 2.0,
            value=lactate,
            threshold="> 2.0 mmol/L",
            clinical_reason="Hyperlactatemia (tissue hypoperfusion, Sepsis-3)",
        ))
        
        # TW-MENTAL: GCS / Mental status
        gcs = latest[FEAT_IDX["gcs_total"]]
        tripwires.append(TripwireResult(
            name="TW-MENTAL",
            triggered=gcs < self.config.gcs_threshold,
            value=gcs,
            threshold=f"GCS < {self.config.gcs_threshold}",
            clinical_reason="Altered mental status",
        ))
        
        # Count active tripwires
        active = [tw for tw in tripwires if tw.triggered]
        n_active = len(active)
        
        # Check for any single extreme vital that warrants immediate CRITICAL
        has_extreme = (
            hr > 150               # Severe tachycardia
            or hr < 40             # Severe bradycardia
            or map_val < 55        # Severe hypotension
            or spo2 < 88           # Severe hypoxemia
            or temp < 34.0         # Severe hypothermia
            or temp > 40.0         # Severe hyperthermia
            or lactate > 4.0       # Severe hyperlactatemia
            or gcs <= 8            # Coma
        )
        
        # Determine override level
        if has_extreme or n_active >= self.config.critical_tripwire_count:
            override_level = "CRITICAL"
        elif n_active >= 1:
            override_level = "AMBER"
        else:
            override_level = "WATCH"
        
        # Build details string
        details_parts = []
        for tw in active:
            details_parts.append(
                f"{tw.name}: {tw.value:.1f} ({tw.threshold}) — {tw.clinical_reason}"
            )
        if has_extreme and n_active < self.config.critical_tripwire_count:
            details_parts.append("EXTREME VITAL VALUE → auto-escalated to CRITICAL")
        details = "; ".join(details_parts) if details_parts else "All vitals within normal range"
        
        return RedTeamAssessment(
            triggered=n_active > 0 or has_extreme,
            active_tripwires=tripwires,
            override_level=override_level,
            n_active=n_active,
            details=details,
        )
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope) of a time series.
        
        Returns:
            Slope in units per hour (positive = increasing)
        """
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            return 0.0
        
        x = np.arange(len(values))[valid]
        y = values[valid]
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-8)
        
        return float(slope)
    
    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize z-scored values back to original scale."""
        result = normalized.copy()
        feature_names = [
            "heart_rate", "sbp", "dbp", "map", "temperature",
            "resp_rate", "spo2", "gcs_total",
            "lactate", "wbc", "creatinine", "platelets"
        ]
        
        for i, feat in enumerate(feature_names):
            if feat in self.norm_stats.get("train_mean", {}):
                mean = self.norm_stats["train_mean"][feat]
                std = self.norm_stats["train_std"][feat]
                result[:, i] = result[:, i] * std + mean
        
        return result
    
    def batch_evaluate(
        self,
        windows: np.ndarray,
    ) -> List[RedTeamAssessment]:
        """Evaluate tripwires on a batch of windows.
        
        Args:
            windows: (N, T, 12) array
        
        Returns:
            List of N RedTeamAssessments
        """
        return [self.evaluate(w) for w in windows]


def test_red_team():
    """Test the Red Team Agent with synthetic scenarios."""
    agent = RedTeamAgent()
    
    print("Red Team Agent — Test Scenarios")
    print("=" * 60)
    
    # Scenario 1: Normal patient
    normal = np.array([
        [80, 120, 70, 85, 37.0, 16, 98, 15, 1.0, 8.0, 0.9, 250],
        [82, 118, 72, 86, 37.0, 15, 97, 15, 1.1, 8.5, 0.9, 245],
        [78, 122, 68, 84, 36.8, 16, 98, 15, 1.0, 7.5, 1.0, 260],
        [81, 119, 71, 85, 36.9, 17, 97, 15, 1.0, 8.0, 0.9, 255],
        [79, 121, 69, 84, 37.1, 16, 98, 15, 0.9, 8.2, 0.9, 248],
        [80, 120, 70, 85, 37.0, 15, 98, 15, 1.0, 8.0, 0.9, 250],
    ], dtype=np.float64)
    
    result = agent.evaluate(normal)
    print(f"\n1. Normal patient:")
    print(f"   Override: {result.override_level}")
    print(f"   Active:   {result.n_active} tripwires")
    print(f"   Details:  {result.details}")
    assert result.override_level == "WATCH"
    
    # Scenario 2: Septic patient (2+ tripwires)
    septic = np.array([
        [85, 115, 65, 80, 37.5, 18, 96, 15, 1.5, 10.0, 1.2, 200],
        [92, 110, 60, 75, 38.0, 20, 95, 15, 2.0, 12.0, 1.5, 180],
        [98, 105, 55, 72, 38.2, 22, 94, 14, 2.5, 14.0, 1.8, 160],
        [105, 100, 52, 68, 38.5, 24, 93, 14, 3.0, 16.0, 2.0, 140],
        [110, 95, 48, 65, 38.8, 26, 92, 13, 3.5, 18.0, 2.2, 120],
        [115, 90, 45, 62, 39.0, 28, 91, 13, 4.0, 20.0, 2.5, 100],
    ], dtype=np.float64)
    
    result = agent.evaluate(septic)
    print(f"\n2. Septic patient (deteriorating):")
    print(f"   Override: {result.override_level}")
    print(f"   Active:   {result.n_active} tripwires")
    print(f"   Details:  {result.details}")
    assert result.override_level == "CRITICAL"
    
    # Scenario 3: Single tripwire (fever only)
    fever = np.array([
        [80, 120, 70, 85, 37.5, 16, 98, 15, 1.0, 8.0, 0.9, 250],
        [82, 118, 72, 86, 37.8, 15, 97, 15, 1.1, 8.5, 0.9, 245],
        [78, 122, 68, 84, 38.0, 16, 98, 15, 1.0, 7.5, 1.0, 260],
        [81, 119, 71, 85, 38.2, 17, 97, 15, 1.0, 8.0, 0.9, 255],
        [79, 121, 69, 84, 38.3, 16, 98, 15, 0.9, 8.2, 0.9, 248],
        [80, 120, 70, 85, 38.5, 15, 98, 15, 1.0, 8.0, 0.9, 250],
    ], dtype=np.float64)
    
    result = agent.evaluate(fever)
    print(f"\n3. Fever only:")
    print(f"   Override: {result.override_level}")
    print(f"   Active:   {result.n_active} tripwires")
    print(f"   Details:  {result.details}")
    assert result.override_level == "AMBER"
    
    print("\n✓ All Red Team Agent tests passed!")


if __name__ == "__main__":
    test_red_team()
