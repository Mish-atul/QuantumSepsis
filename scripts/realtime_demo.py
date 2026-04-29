"""
QuantumSepsis Shield — Real-Time Demo Dashboard
Simulates 3 patient scenarios with live updates every 60 seconds.
"""
import streamlit as st
import numpy as np
import torch
import json
import time
from pathlib import Path
import sys

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
    conformal_data = json.load(open('data/processed/conformal_calibration.json'))
    conformal.q_alpha = conformal_data['q_alpha']
    conformal.calibrated = True
    
    # Red Team
    norm_stats = json.load(open('data/processed/normalization_stats.json'))
    red_team = RedTeamAgent(config.red_team, use_normalized=True, norm_stats=norm_stats)
    
    # Orchestrator
    orchestrator = ConfidenceGatedOrchestrator(config.orchestrator)
    
    return lstm, conformal, red_team, orchestrator

try:
    lstm, conformal, red_team, orchestrator = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Make sure you have run the pipeline and have checkpoints/lstm_best.pt and data/processed/conformal_calibration.json")
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

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

if auto_refresh:
    time.sleep(60)
    st.rerun()

# Generate current window
hour = int(time.time() / 60) % 24  # Simulate time progression
window = generate_patient_window(scenario, hour)

# Normalize using TRAINING SET statistics (not window's own stats!)
norm_stats = json.load(open('data/processed/normalization_stats.json'))
feature_names = ['heart_rate', 'sbp', 'dbp', 'map', 'temperature', 'resp_rate', 
                 'spo2', 'gcs_total', 'lactate', 'wbc', 'creatinine', 'platelets']
means = np.array([norm_stats['train_mean'][f] for f in feature_names])
stds = np.array([norm_stats['train_std'][f] for f in feature_names])
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

# Display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Score", f"{risk_score:.3f}", delta=None)
    st.metric("Confidence", f"{confidence:.2f}", delta=None)

with col2:
    alert_color = {
        "WATCH": "🟢",
        "AMBER": "🟡",
        "CRITICAL": "🔴",
        "FAST-TRACK": "⚡"
    }
    st.metric("Alert Level", f"{alert_color.get(decision.alert_level, '')} {decision.alert_level}")
    st.metric("Red Team", f"{red_team_result.n_active} tripwires")

with col3:
    st.metric("Conformal Interval", f"[{lower:.3f}, {upper:.3f}]")
    st.metric("Fast-Tracked", "Yes" if decision.fast_tracked else "No")

# Vital signs chart
st.subheader("📊 Vital Signs (Last 6 Hours)")
import pandas as pd

hours = list(range(-5, 1))
df = pd.DataFrame({
    'Hour': hours,
    'Heart Rate': window[:, 0],
    'MAP': window[:, 3],
    'Temperature': window[:, 4],
    'Resp Rate': window[:, 5],
    'SpO2': window[:, 6],
})

# Use Streamlit's native line chart
st.line_chart(df.set_index('Hour'))

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
    st.write(f"- {action}")

# Reasoning
with st.expander("🧠 Decision Reasoning"):
    st.write(decision.reasoning)

# Footer
st.markdown("---")
st.markdown("**QuantumSepsis Shield** | Phase 3 Demo | MIMIC-IV v3.1")
