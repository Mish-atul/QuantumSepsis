"""
QuantumSepsis Shield — FastAPI Backend (Phase 2)
=================================================
Multimodal pipeline: Vitals + Demographics + Chest X-Ray + LLM Reasoning.

Endpoints:
  GET  /health   — readiness probe
  POST /predict  — full pipeline: vitals → LSTM+XGBoost → Conformal → RedTeam
                   → CXR analysis → LLM narrative → orchestrator decision
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
    description="Real-time sepsis risk prediction using LSTM + XGBoost ensemble with CXR + LLM",
    version="2.0.0",
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
    # Phase 2: Demographics
    age: float = 55.0
    gender: str = "M"
    # Phase 2: CXR image (base64-encoded JPEG/PNG, optional)
    cxr_image_base64: Optional[str] = None


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
    # Phase 2 fields
    demographics: Optional[Dict] = None
    cxr_findings: Optional[Dict] = None
    clinical_narrative: Optional[str] = None


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
        "version": "2.0.0-phase2",
    }


@app.post("/predict")
def predict(vitals: VitalInputs):
    """Run the full multimodal pipeline on the given vitals.

    Pipeline: vitals → LSTM+XGBoost → Conformal → RedTeam → CXR → LLM → Orchestrator
    """
    window = build_window(vitals)
    result = runtime.predict_one(window)

    # ── Core pipeline (unchanged from Phase 1) ─────────────────
    red_team = result["red_team"]
    decision = result["decision"]

    # Build tripwire list — all 7 vital-sign tripwires
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

    # ── Phase 2: CXR Analysis ──────────────────────────────────
    cxr_data = None
    cxr_risk_modifier = 0.0
    if vitals.cxr_image_base64:
        try:
            from src.models.cxr_encoder import (
                analyze_cxr_base64,
                get_sepsis_relevant_findings,
            )

            raw_scores = analyze_cxr_base64(vitals.cxr_image_base64)
            cxr_data = get_sepsis_relevant_findings(raw_scores)
            cxr_risk_modifier = cxr_data["risk_modifier"]

            # Add CXR imaging tripwires
            cxr_tripwires = runtime.red_team.evaluate_cxr(cxr_data)
            for tw in cxr_tripwires:
                tripwire_list.append({
                    "name": tw.name,
                    "triggered": tw.triggered,
                    "value": float(tw.value),
                    "threshold": tw.threshold,
                    "reason": tw.clinical_reason,
                })
            logger.info(
                "CXR analysis: modifier=+%.3f, findings=%d",
                cxr_risk_modifier,
                len(cxr_data.get("findings", [])),
            )
        except ImportError:
            logger.warning("torchxrayvision not installed — skipping CXR analysis")
            cxr_data = {"error": "torchxrayvision not installed"}
        except Exception as e:
            logger.warning("CXR analysis failed: %s", e)
            cxr_data = {"error": str(e)}

    # ── Phase 2: Demographics context ──────────────────────────
    demographics = {
        "age": vitals.age,
        "gender": vitals.gender,
        "age_risk_note": (
            "Elderly (>65) — increased sepsis mortality risk"
            if vitals.age > 65
            else "Standard age range"
        ),
    }

    # ── Adjust risk score with CXR modifier ────────────────────
    final_risk = min(1.0, float(result["risk_score"]) + cxr_risk_modifier)

    # ── Phase 2: LLM Clinical Narrative ────────────────────────
    clinical_narrative = None
    try:
        from src.agents.clinical_reasoning import generate_clinical_narrative

        narrative_context = {
            "age": vitals.age,
            "gender": vitals.gender,
            "heart_rate": vitals.heart_rate,
            "map": vitals.map,
            "temperature": vitals.temperature,
            "resp_rate": vitals.resp_rate,
            "spo2": vitals.spo2,
            "gcs_total": vitals.gcs_total,
            "lactate": vitals.lactate,
            "wbc": vitals.wbc,
            "creatinine": vitals.creatinine,
            "platelets": vitals.platelets,
            "risk_score": final_risk,
            "conf_lower": float(result["conformal_lower"]),
            "conf_upper": float(result["conformal_upper"]),
            "active_tripwires": [
                tw["name"] for tw in tripwire_list if tw["triggered"]
            ],
            "alert_level": (
                "FAST-TRACK" if decision.fast_tracked else decision.alert_level
            ),
            "cxr_findings": (
                cxr_data["summary"]
                if cxr_data and "summary" in cxr_data
                else "Not available"
            ),
        }
        clinical_narrative = generate_clinical_narrative(narrative_context)
    except Exception as e:
        logger.warning("LLM narrative generation failed: %s", e)
        clinical_narrative = None

    # ── Build response ─────────────────────────────────────────
    action_strings = [
        f"[{a.time_sensitivity}] {a.description}" for a in decision.actions
    ]

    alert_level = (
        "FAST-TRACK" if decision.fast_tracked else decision.alert_level
    )
    n_active = sum(1 for tw in tripwire_list if tw["triggered"])

    return {
        "risk_score": round(final_risk, 4),
        "lstm_score": round(float(result["lstm_score"]), 4),
        "xgb_score": (
            round(float(result["xgb_score"]), 4)
            if result["xgb_score"] is not None
            else None
        ),
        "confidence": round(float(result["confidence"]), 4),
        "conformal_interval": [
            round(float(result["conformal_lower"]), 4),
            round(float(result["conformal_upper"]), 4),
        ],
        "alert_level": alert_level,
        "fast_tracked": bool(decision.fast_tracked),
        "tripwires": tripwire_list,
        "n_active_tripwires": n_active,
        "has_extreme": bool(has_extreme),
        "reasoning": str(decision.reasoning),
        "actions": action_strings,
        "backend": str(result["status"].backend_mode),
        # Phase 2 fields
        "demographics": demographics,
        "cxr_findings": cxr_data,
        "clinical_narrative": clinical_narrative,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
