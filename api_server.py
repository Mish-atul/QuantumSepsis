"""
QuantumSepsis Shield — FastAPI Backend
======================================
Exposes the real LSTM + XGBoost + Conformal + Red Team pipeline
via a JSON REST API so the Vercel React UI can call it.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.demo_runtime import DemoInferenceRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Initialize FastAPI ──────────────────────────────────────────
app = FastAPI(
    title="QuantumSepsis Shield API",
    description="Real-time sepsis risk prediction using LSTM + XGBoost ensemble",
    version="1.0.0",
)

# Allow Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load the real model at startup ──────────────────────────────
runtime = None


@app.on_event("startup")
def load_model():
    global runtime
    logger.info("Loading QuantumSepsis pipeline (LSTM + XGBoost + Conformal)...")
    runtime = DemoInferenceRuntime()
    logger.info(f"Pipeline loaded. Backend mode: {runtime.status.backend_mode}")
    if runtime.status.warnings:
        for w in runtime.status.warnings:
            logger.warning(w)


# ── Request / Response Models ───────────────────────────────────
class VitalInputs(BaseModel):
    heart_rate: float = 75.0
    map: float = 85.0
    temperature: float = 37.0
    resp_rate: float = 14.0
    spo2: float = 98.0
    gcs_total: float = 15.0
    lactate: float = 1.0
    wbc: float = 8.0
    creatinine: float = 0.9
    platelets: float = 220.0


class TripwireItem(BaseModel):
    name: str
    triggered: bool
    value: float
    threshold: str
    reason: str


class PredictionResponse(BaseModel):
    risk_score: float
    lstm_score: float
    xgb_score: Optional[float] = None
    confidence: float
    conformal_interval: List[float]
    alert_level: str
    fast_tracked: bool
    tripwires: List[TripwireItem]
    n_active_tripwires: int
    has_extreme: bool
    reasoning: str
    actions: List[str]
    backend: str


# ── Helper: build 6x12 window from single vitals snapshot ──────
def build_window(v: VitalInputs) -> np.ndarray:
    """Build a 6-step window from user-supplied vitals (same as Streamlit)."""
    def ramp(target: float, delta: float = 0.1) -> np.ndarray:
        start = target * (1 - delta)
        return np.linspace(start, target, 6)

    map_v = v.map
    sbp = map_v + 40
    dbp = map_v - 20

    return np.stack(
        [
            ramp(v.heart_rate),
            ramp(sbp),
            ramp(dbp),
            ramp(map_v),
            ramp(v.temperature, 0.005),
            ramp(v.resp_rate),
            ramp(v.spo2, 0.02),
            np.ones(6) * v.gcs_total,
            ramp(v.lactate, 0.15),
            ramp(v.wbc, 0.08),
            ramp(v.creatinine, 0.08),
            ramp(v.platelets, 0.05),
        ],
        axis=1,
    )


# ── Endpoints ───────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": runtime is not None,
        "backend_mode": runtime.status.backend_mode if runtime else None,
    }


@app.post("/predict")
def predict(vitals: VitalInputs):
    """Run the real LSTM + XGBoost pipeline on the given vitals."""
    window = build_window(vitals)
    result = runtime.predict_one(window)

    # Extract red team data
    red_team = result["red_team"]
    decision = result["decision"]

    # Build tripwire list — include ALL 7 tripwires (both triggered and not)
    tripwire_list = []
    for tw in red_team.active_tripwires:
        tripwire_list.append({
            "name": tw.name,
            "triggered": tw.triggered,
            "value": float(tw.value),
            "threshold": tw.threshold,
            "reason": tw.clinical_reason,
        })

    # Check for extreme values
    v = vitals
    has_extreme = (
        v.heart_rate > 150 or v.heart_rate < 40 or
        v.map < 55 or v.spo2 < 88 or
        v.temperature < 34.0 or v.temperature > 40.0 or
        v.lactate > 4.0 or v.gcs_total <= 8
    )

    # Build action strings
    action_strings = [
        f"[{a.time_sensitivity}] {a.description}"
        for a in decision.actions
    ]

    # Determine display alert level
    alert_level = "FAST-TRACK" if decision.fast_tracked else decision.alert_level

    return {
        "risk_score": round(float(result["risk_score"]), 4),
        "lstm_score": round(float(result["lstm_score"]), 4),
        "xgb_score": round(float(result["xgb_score"]), 4) if result["xgb_score"] is not None else None,
        "confidence": round(float(result["confidence"]), 4),
        "conformal_interval": [round(float(result["conformal_lower"]), 4), round(float(result["conformal_upper"]), 4)],
        "alert_level": alert_level,
        "fast_tracked": bool(decision.fast_tracked),
        "tripwires": tripwire_list,
        "n_active_tripwires": int(red_team.n_active),
        "has_extreme": bool(has_extreme),
        "reasoning": str(decision.reasoning),
        "actions": action_strings,
        "backend": str(result["status"].backend_mode),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
