"""
QuantumSepsis Shield — Real-Time Demo Dashboard
Simulates 3 patient scenarios with live updates every 60 seconds.
"""
import streamlit as st
import numpy as np
import torch
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm import SepsisLSTM
from src.models.conformal import ConformalSepsisPredictor
from src.agents.red_team import RedTeamAgent
from src.agents.orchestrator import ConfidenceGatedOrchestrator
from src.config import get_default_config

# Page config
st.set_page_config(
    page_title="QuantumSepsis Shield",
    page_icon="🛡️",
    layout="wide"
)

st.markdown(
        """
<style>
:root {
    --bg: #0b1220;
    --card: rgba(255, 255, 255, 0.07);
    --border: rgba(255, 255, 255, 0.12);
    --text: #e6edf7;
    --muted: #9fb2d0;
    --green: #18d26e;
    --amber: #ffb02e;
    --red: #ff4d4f;
}

.stApp {
    background: radial-gradient(1200px 600px at 10% 10%, #1b2a4d 0%, #0b1220 45%, #070b14 100%);
    color: var(--text);
}

.glass {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
}

.alert-banner {
    border-radius: 16px;
    padding: 16px 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    border: 1px solid var(--border);
}

.alert-watch { background: rgba(24, 210, 110, 0.2); }
.alert-amber { background: rgba(255, 176, 46, 0.25); }
.alert-critical { background: rgba(255, 77, 79, 0.25); }
.alert-fast { background: rgba(255, 77, 79, 0.35); }

.pulse {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 77, 79, 0.6); }
    70% { box-shadow: 0 0 0 14px rgba(255, 77, 79, 0.0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 77, 79, 0.0); }
}

.muted { color: var(--muted); }
</style>
""",
        unsafe_allow_html=True,
)

# Load models (cached)
@st.cache_resource
def load_models():
    config = get_default_config()
    
    # LSTM
    lstm = SepsisLSTM(config.lstm)
    checkpoint = torch.load('checkpoints/lstm_best.pt', map_location='cpu')
    lstm.load_state_dict(checkpoint['model_state_dict'])
    lstm.eval()
    
    # Conformal
    conformal = ConformalSepsisPredictor(config.conformal)
    conformal_path = Path('data/processed/conformal_calibration.json')
    if conformal_path.exists():
        conformal_data = json.loads(conformal_path.read_text())
        conformal.q_alpha = conformal_data['q_alpha']
    else:
        conformal.q_alpha = 0.3923025131
    conformal.calibrated = True
    
    # Red Team (demo uses raw clinical values, not z-normalized)
    red_team = RedTeamAgent(config.red_team)
    
    # Orchestrator
    orchestrator = ConfidenceGatedOrchestrator(config.orchestrator)
    
    return lstm, conformal, red_team, orchestrator

try:
    lstm, conformal, red_team, orchestrator = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Make sure you have run the pipeline and have checkpoints/lstm_best.pt")
    models_loaded = False

# Simulate 3 patient scenarios
def generate_patient_window(scenario, hour):
    """Generate synthetic 6-hour window for demo."""
    np.random.seed(hour)  # Deterministic for demo
    
    if scenario == "normal":
        # Stable vitals
        hr = np.random.normal(75, 5, 6)
        map_val = np.random.normal(85, 5, 6)
        temp = np.random.normal(37.0, 0.2, 6)
        rr = np.random.normal(16, 2, 6)
        spo2 = np.random.normal(98, 1, 6)
        gcs = np.ones(6) * 15
        
    elif scenario == "slow_deterioration":
        # Gradual decline
        hr = np.linspace(80, 95, 6) + np.random.normal(0, 3, 6)
        map_val = np.linspace(85, 68, 6) + np.random.normal(0, 3, 6)
        temp = np.linspace(37.0, 38.5, 6) + np.random.normal(0, 0.2, 6)
        rr = np.linspace(16, 22, 6) + np.random.normal(0, 1, 6)
        spo2 = np.linspace(98, 94, 6) + np.random.normal(0, 1, 6)
        gcs = np.linspace(15, 13, 6)
        
    else:  # rapid_sepsis
        # Rapid deterioration
        hr = np.array([82, 88, 95, 105, 115, 120]) + np.random.normal(0, 3, 6)
        map_val = np.array([85, 80, 72, 65, 60, 58]) + np.random.normal(0, 2, 6)
        temp = np.array([37.2, 37.8, 38.5, 39.0, 39.2, 39.5]) + np.random.normal(0, 0.1, 6)
        rr = np.array([18, 20, 24, 28, 30, 32]) + np.random.normal(0, 1, 6)
        spo2 = np.array([98, 96, 94, 91, 89, 87]) + np.random.normal(0, 1, 6)
        gcs = np.array([15, 15, 14, 13, 12, 11])
    
    # Build 12-feature window (simplified - add other features as needed)
    sbp = map_val + 40  # Approximate SBP from MAP
    dbp = map_val - 20  # Approximate DBP from MAP
    lactate = np.ones(6) * 1.5
    wbc = np.ones(6) * 10
    creatinine = np.ones(6) * 1.0
    platelets = np.ones(6) * 200
    
    window = np.stack([hr, sbp, dbp, map_val, temp, rr, spo2, gcs,
                       lactate, wbc, creatinine, platelets], axis=1)
    return window


def build_risk_gauge(risk_score: float) -> go.Figure:
    """Create a semicircular risk gauge."""
    value = max(0.0, min(1.0, risk_score)) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4d4f"},
                "steps": [
                    {"range": [0, 30], "color": "#1f9d55"},
                    {"range": [30, 60], "color": "#f1c40f"},
                    {"range": [60, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf7"},
    )
    return fig


def status_badge(value: float, low: float, high: float) -> str:
    if value < low:
        return "🔴 LOW"
    if value > high:
        return "⚠️ HIGH"
    return "✅ NORMAL"

# Main dashboard
st.title("🛡️ QuantumSepsis Shield — Real-Time Demo")
st.markdown("**Adversarially-Safe Quantum-Classical System for Early Sepsis Detection**")

if not models_loaded:
    st.stop()

# Sidebar: scenario selection
st.sidebar.header("Patient Scenarios")
scenario = st.sidebar.selectbox(
    "Select Patient",
    ["normal", "slow_deterioration", "rapid_sepsis"],
    format_func=lambda x: {
        "normal": "👤 Patient A: Stable",
        "slow_deterioration": "⚠️ Patient B: Slow Deterioration",
        "rapid_sepsis": "🚨 Patient C: Rapid Sepsis"
    }[x]
)

scenario_summary = {
    "normal": "Stable vitals with no infection markers.",
    "slow_deterioration": "Gradual hypotension, rising fever, and tachypnea.",
    "rapid_sepsis": "Fast decompensation: fever, tachycardia, hypotension, and hypoxia.",
}
scenario_features = {
    "normal": ["HR 70-80", "MAP ~85", "Temp ~37.0", "SpO2 97-99"],
    "slow_deterioration": ["HR drifting up", "MAP dropping", "Temp rising", "RR > 20"],
    "rapid_sepsis": ["HR > 110", "MAP < 65", "Temp > 39", "SpO2 < 90"],
}

st.sidebar.markdown("### Scenario Summary")
st.sidebar.write(scenario_summary[scenario])
st.sidebar.markdown("### Key Clinical Features")
st.sidebar.write("\n".join([f"- {item}" for item in scenario_features[scenario]]))
st.sidebar.markdown("### Model Architecture")
st.sidebar.code(
    """Vitals → Windowing → BiLSTM → Embedding → Quantum Kernel
BiLSTM → Conformal Prediction
Vitals → Red Team Agent
All signals → Orchestrator → Alert"""
)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

if auto_refresh:
    time.sleep(60)
    st.rerun()

# Generate current window
hour = int(time.time() / 60) % 24  # Simulate time progression
window = generate_patient_window(scenario, hour)

# Normalize using TRAINING SET statistics (not window's own stats!)
feature_names = ['heart_rate', 'sbp', 'dbp', 'map', 'temperature', 'resp_rate', 
                 'spo2', 'gcs_total', 'lactate', 'wbc', 'creatinine', 'platelets']
norm_stats_path = Path('data/processed/normalization_stats.json')
if norm_stats_path.exists():
    norm_stats = json.loads(norm_stats_path.read_text())
    means = np.array([norm_stats['train_mean'][f] for f in feature_names])
    stds = np.array([norm_stats['train_std'][f] for f in feature_names])
else:
    means = np.zeros(len(feature_names), dtype=float)
    stds = np.ones(len(feature_names), dtype=float)
window_norm = (window - means) / (stds + 1e-8)

# Run inference
with torch.no_grad():
    window_tensor = torch.FloatTensor(window_norm).unsqueeze(0)
    outputs = lstm(window_tensor)
    risk_score = outputs['risk_score'].item()

# Conformal prediction
lower = max(0, risk_score - conformal.q_alpha)
upper = min(1, risk_score + conformal.q_alpha)
confidence = 1.0 - (upper - lower)

# Red Team
red_team_result = red_team.evaluate(window)

# Orchestrator
decision = orchestrator.decide(
    risk_score=risk_score,
    conformal_lower=lower,
    conformal_upper=upper,
    red_team=red_team_result
)

# Alert banner
banner_label = "FAST-TRACK" if decision.fast_tracked else decision.alert_level
banner_display = "⚡ FAST-TRACK" if decision.fast_tracked else decision.alert_level
banner_class = {
    "WATCH": "alert-watch",
    "AMBER": "alert-amber",
    "CRITICAL": "alert-critical",
    "FAST-TRACK": "alert-fast",
}[banner_label]
pulse_class = "pulse" if banner_label in {"CRITICAL", "FAST-TRACK"} else ""
st.markdown(
    f"""
<div class="alert-banner {banner_class} {pulse_class}">
    ALERT: {banner_display} · Risk {risk_score:.3f} · Confidence {confidence:.2f}
</div>
""",
    unsafe_allow_html=True,
)

left, mid, right = st.columns([1.1, 1.2, 1.7])

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.plotly_chart(build_risk_gauge(risk_score), use_container_width=True)
    st.markdown(
        f"<div class='muted'>Conformal interval: [{lower:.3f}, {upper:.3f}]</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with mid:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Decision Summary")
    st.write(f"**Alert Level:** {banner_label}")
    st.write(f"**Red Team Tripwires:** {red_team_result.n_active}")
    st.write(f"**Fast-Tracked:** {'Yes' if decision.fast_tracked else 'No'}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Vital Signs Snapshot")
    latest = window[-1]
    vitals = [
        ("Heart Rate", latest[0], "bpm", 60, 100),
        ("MAP", latest[3], "mmHg", 70, 105),
        ("Temperature", latest[4], "°C", 36.0, 38.3),
        ("Resp Rate", latest[5], "br/min", 12, 20),
        ("SpO2", latest[6], "%", 95, 100),
        ("GCS", latest[7], "", 14, 15),
    ]
    rows = []
    for name, value, unit, low, high in vitals:
        rows.append({
            "Vital": name,
            "Current": f"{value:.1f} {unit}".strip(),
            "Normal Range": f"{low}-{high} {unit}".strip(),
            "Status": status_badge(value, low, high),
        })
    vitals_df = pd.DataFrame(rows)
    st.dataframe(vitals_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Risk history
if "risk_history" not in st.session_state:
    st.session_state["risk_history"] = {}
history = st.session_state["risk_history"].setdefault(scenario, [])
history.append({
    "time": datetime.now().strftime("%H:%M:%S"),
    "risk": risk_score,
    "confidence": confidence,
})
history = history[-60:]
st.session_state["risk_history"][scenario] = history

st.subheader("📈 Risk Timeline")
history_df = pd.DataFrame(history)
if len(history_df) > 1:
    st.line_chart(history_df.set_index("time")["risk"])
else:
    st.caption("Collecting data for timeline...")

# Vital signs chart
st.subheader("📊 Vital Signs (Last 6 Hours)")
hours = list(range(-5, 1))
df = pd.DataFrame({
    "Hour": hours,
    "Heart Rate": window[:, 0],
    "MAP": window[:, 3],
    "Temperature": window[:, 4],
    "Resp Rate": window[:, 5],
    "SpO2": window[:, 6],
})
st.line_chart(df.set_index("Hour"))

# Tripwire panel
st.subheader("🚨 Clinical Tripwires")
for tw in red_team_result.active_tripwires:
    if tw.triggered:
        st.error(f"**{tw.name}**: {tw.value:.1f} ({tw.threshold}) — {tw.clinical_reason}")
    else:
        st.success(f"**{tw.name}**: {tw.value:.1f} (Normal)")

# Clinical actions
st.subheader("📋 Recommended Actions")
for action in decision.actions:
    st.write(
        f"- **[{action.action_type}]** {action.description} "
        f"*(Priority: {action.priority}, {action.time_sensitivity})*"
    )

# Reasoning
with st.expander("🧠 Decision Reasoning"):
    st.write(decision.reasoning)

# Footer
st.markdown("---")
st.markdown("**QuantumSepsis Shield** | Phase 3 Demo | MIMIC-IV v3.1")
