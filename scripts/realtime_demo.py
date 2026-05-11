"""
QuantumSepsis Shield - Real-Time Demo Dashboard.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.demo_runtime import DemoInferenceRuntime

st.set_page_config(page_title="QuantumSepsis Shield", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg: #0b1220;
    --card: rgba(255, 255, 255, 0.07);
    --border: rgba(255, 255, 255, 0.12);
    --text: #e6edf7;
    --muted: #9fb2d0;
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
.muted { color: var(--muted); }
.alert-banner {
    border-radius: 16px;
    padding: 16px 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.12);
}
.alert-watch { background: rgba(24, 210, 110, 0.2); }
.alert-amber { background: rgba(255, 176, 46, 0.25); }
.alert-critical { background: rgba(255, 77, 79, 0.25); }
.alert-fast { background: rgba(255, 77, 79, 0.35); }
</style>
""",
    unsafe_allow_html=True,
)

SCENARIOS = {
    "normal": {
        "label": "Patient A: Stable ICU",
        "desc": "Post-operative day 2 with stable vitals and no infection markers.",
    },
    "early_warning": {
        "label": "Patient B: Early Warning",
        "desc": "Subtle warning signs with mild tachycardia, fever, and rising lactate.",
    },
    "slow_deterioration": {
        "label": "Patient C: Slow Deterioration",
        "desc": "Progressive decline over 6 hours with hypotension and tachypnea.",
    },
    "rapid_sepsis": {
        "label": "Patient D: Rapid Sepsis",
        "desc": "Acute decompensation consistent with severe sepsis progression.",
    },
    "hypothermic_sepsis": {
        "label": "Patient E: Hypothermic Sepsis",
        "desc": "Cold sepsis presentation with hypothermia and organ dysfunction.",
    },
    "custom": {
        "label": "Custom Input",
        "desc": "Enter live values for a manual demo run.",
    },
}

FEATURE_NAMES = [
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "temperature",
    "resp_rate",
    "spo2",
    "gcs_total",
    "lactate",
    "wbc",
    "creatinine",
    "platelets",
]


@st.cache_resource
def load_runtime(_version: int = 2) -> DemoInferenceRuntime:
    """Load the inference runtime. Bump _version to bust cache after code changes."""
    return DemoInferenceRuntime()


def init_state() -> None:
    if "sim_time" not in st.session_state:
        st.session_state.sim_time = datetime.now().replace(second=0, microsecond=0)
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "cycle_id" not in st.session_state:
        st.session_state.cycle_id = 0
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = {}
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "latest_window" not in st.session_state:
        st.session_state.latest_window = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = "Manual Instant"


def generate_patient_window(scenario: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = lambda s: rng.normal(0, s, 6)

    if scenario == "normal":
        hr = rng.normal(75, 3, 6)
        map_val = rng.normal(85, 3, 6)
        temp = rng.normal(36.8, 0.15, 6)
        rr = rng.normal(14, 1.5, 6)
        spo2 = np.clip(rng.normal(98, 0.5, 6), 92, 100)
        gcs = np.ones(6) * 15
        lactate = rng.normal(1.0, 0.1, 6)
        wbc = rng.normal(8, 1, 6)
        creat = rng.normal(0.9, 0.05, 6)
        plts = rng.normal(220, 15, 6)
    elif scenario == "early_warning":
        hr = np.linspace(82, 92, 6) + noise(2)
        map_val = np.linspace(82, 75, 6) + noise(2)
        temp = np.linspace(37.2, 38.0, 6) + noise(0.1)
        rr = np.linspace(16, 20, 6) + noise(1)
        spo2 = np.clip(np.linspace(97, 95, 6) + noise(0.5), 88, 100)
        gcs = np.ones(6) * 15
        lactate = np.linspace(1.2, 1.8, 6) + noise(0.1)
        wbc = np.linspace(9, 12, 6) + noise(0.5)
        creat = np.linspace(1.0, 1.2, 6) + noise(0.03)
        plts = np.linspace(200, 185, 6) + noise(8)
    elif scenario == "slow_deterioration":
        hr = np.linspace(80, 100, 6) + noise(3)
        map_val = np.linspace(82, 65, 6) + noise(2)
        temp = np.linspace(37.2, 38.5, 6) + noise(0.15)
        rr = np.linspace(16, 24, 6) + noise(1)
        spo2 = np.clip(np.linspace(97, 93, 6) + noise(0.8), 84, 100)
        gcs = np.array([15, 15, 15, 14, 14, 13], dtype=float)
        lactate = np.linspace(1.5, 3.5, 6) + noise(0.2)
        wbc = np.linspace(10, 15, 6) + noise(1)
        creat = np.linspace(1.0, 1.6, 6) + noise(0.05)
        plts = np.linspace(200, 150, 6) + noise(10)
    elif scenario == "rapid_sepsis":
        hr = np.array([82, 92, 105, 115, 122, 128]) + noise(3)
        map_val = np.array([82, 75, 68, 62, 58, 55]) + noise(2)
        temp = np.array([37.2, 37.8, 38.5, 39.0, 39.3, 39.5]) + noise(0.1)
        rr = np.array([18, 22, 26, 30, 32, 34]) + noise(1)
        spo2 = np.clip(np.array([97, 95, 93, 90, 88, 86]) + noise(0.5), 80, 100)
        gcs = np.array([15, 15, 14, 13, 12, 11], dtype=float)
        lactate = np.array([1.5, 2.2, 3.0, 4.5, 5.5, 6.5]) + noise(0.2)
        wbc = np.array([10, 13, 16, 18, 20, 22]) + noise(1)
        creat = np.array([1.0, 1.2, 1.5, 1.9, 2.3, 2.8]) + noise(0.05)
        plts = np.array([200, 175, 150, 120, 95, 75]) + noise(5)
    elif scenario == "hypothermic_sepsis":
        hr = np.array([65, 62, 58, 55, 52, 50]) + noise(2)
        map_val = np.array([72, 68, 64, 60, 58, 55]) + noise(2)
        temp = np.array([36.0, 35.5, 35.0, 34.8, 34.5, 34.2]) + noise(0.1)
        rr = np.array([18, 20, 22, 24, 26, 28]) + noise(1)
        spo2 = np.clip(np.array([96, 95, 93, 91, 90, 88]) + noise(0.5), 80, 100)
        gcs = np.array([14, 13, 13, 12, 11, 11], dtype=float)
        lactate = np.array([2.0, 2.8, 3.5, 4.2, 5.0, 5.5]) + noise(0.2)
        wbc = np.array([3, 3.5, 4, 4.5, 5, 5.5]) + noise(0.3)
        creat = np.array([1.2, 1.5, 1.8, 2.0, 2.3, 2.5]) + noise(0.05)
        plts = np.array([180, 150, 120, 100, 80, 65]) + noise(5)
    else:
        hr = np.ones(6) * 75
        map_val = np.ones(6) * 85
        temp = np.ones(6) * 37.0
        rr = np.ones(6) * 14
        spo2 = np.ones(6) * 98
        gcs = np.ones(6) * 15
        lactate = np.ones(6) * 1.0
        wbc = np.ones(6) * 8
        creat = np.ones(6) * 0.9
        plts = np.ones(6) * 220

    sbp = map_val + 40
    dbp = map_val - 20
    return np.stack([hr, sbp, dbp, map_val, temp, rr, spo2, gcs, lactate, wbc, creat, plts], axis=1)


def generate_custom_window(values: Dict[str, float]) -> np.ndarray:
    """Build a 6-step window from user-supplied vitals.
    SBP and DBP are auto-derived from MAP."""
    def ramp(target: float, delta: float = 0.1) -> np.ndarray:
        start = target * (1 - delta)
        return np.linspace(start, target, 6)

    map_v = values["map"]
    sbp = map_v + 40   # standard approximation
    dbp = map_v - 20

    return np.stack(
        [
            ramp(values["heart_rate"]),
            ramp(sbp),
            ramp(dbp),
            ramp(map_v),
            ramp(values["temperature"], 0.005),
            ramp(values["resp_rate"]),
            ramp(values["spo2"], 0.02),
            np.ones(6) * values["gcs_total"],
            ramp(values["lactate"], 0.15),
            ramp(values["wbc"], 0.08),
            ramp(values["creatinine"], 0.08),
            ramp(values["platelets"], 0.05),
        ],
        axis=1,
    )


def build_risk_gauge(risk_score: float) -> go.Figure:
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
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def append_history(scenario: str, timestamp_str: str, result: Dict) -> None:
    history = st.session_state.risk_history.setdefault(scenario, [])
    history.append(
        {
            "time": timestamp_str,
            "risk": result["risk_score"],
            "confidence": result["confidence"],
        }
    )
    st.session_state.risk_history[scenario] = history[-100:]


def run_prediction(runtime: DemoInferenceRuntime, scenario: str, window: np.ndarray, timestamp_str: str) -> None:
    result = runtime.predict_one(window)
    st.session_state.latest_result = result
    st.session_state.latest_window = window
    append_history(scenario, timestamp_str, result)


def status_badge(value: float, low: float, high: float) -> str:
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"


def render_dashboard(result: Dict, window: np.ndarray, scenario: str, mode: str, timeline_label: str) -> None:
    decision = result["decision"]
    red_team_result = result["red_team"]
    status = result["status"]
    risk_score = result["risk_score"]
    confidence = result["confidence"]
    lower = result["conformal_lower"]
    upper = result["conformal_upper"]

    banner_level = "FAST-TRACK" if decision.fast_tracked else decision.alert_level
    banner_class = {
        "WATCH": "alert-watch",
        "AMBER": "alert-amber",
        "CRITICAL": "alert-critical",
        "FAST-TRACK": "alert-fast",
    }[banner_level]

    st.markdown(
        f"""
<div class="alert-banner {banner_class}">
  ALERT: {banner_level} | Risk {risk_score:.3f} | Confidence {confidence:.2f}
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(f"Backend: `{status.backend_mode}` | Mode: `{mode}` | Timeline: {timeline_label}")
    for warning in status.warnings:
        st.warning(warning)

    left, mid, right = st.columns([1.1, 1.2, 1.7])
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.plotly_chart(build_risk_gauge(risk_score), use_container_width=True)
        st.markdown(f"<div class='muted'>Conformal interval: [{lower:.3f}, {upper:.3f}]</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Decision Summary")
        st.write(f"**Alert Level:** {banner_level}")
        st.write(f"**Red Team Tripwires:** {red_team_result.n_active}")
        st.write(f"**Fast-Tracked:** {'Yes' if decision.fast_tracked else 'No'}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**LSTM Score:** {result['lstm_score']:.3f}")
        if result["xgb_score"] is not None:
            st.write(f"**XGBoost Score:** {result['xgb_score']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Vital Signs Snapshot")
        latest = window[-1]
        rows = [
            ("Heart Rate", latest[0], "bpm", 60, 100),
            ("MAP", latest[3], "mmHg", 70, 105),
            ("Temperature", latest[4], "C", 36.0, 38.3),
            ("Resp Rate", latest[5], "br/min", 12, 20),
            ("SpO2", latest[6], "%", 95, 100),
            ("GCS", latest[7], "", 14, 15),
            ("Lactate", latest[8], "mmol/L", 0.5, 2.0),
        ]
        vitals_df = pd.DataFrame(
            [
                {
                    "Vital": name,
                    "Current": f"{value:.1f} {unit}".strip(),
                    "Normal Range": f"{low}-{high} {unit}".strip(),
                    "Status": status_badge(value, low, high),
                }
                for name, value, unit, low, high in rows
            ]
        )
        st.dataframe(vitals_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Risk Timeline")
    history_df = pd.DataFrame(st.session_state.risk_history.get(scenario, []))
    if len(history_df) > 1:
        st.line_chart(history_df.set_index("time")[["risk", "confidence"]])
    else:
        st.caption("Run more cycles to build timeline.")

    st.subheader("Vital Signs (Last 6 Hours)")
    hours = list(range(-5, 1))
    chart_df = pd.DataFrame(
        {
            "Hour": hours,
            "Heart Rate": window[:, 0],
            "MAP": window[:, 3],
            "Temperature": window[:, 4],
            "Resp Rate": window[:, 5],
            "SpO2": window[:, 6],
        }
    )
    st.line_chart(chart_df.set_index("Hour"))

    st.subheader("Clinical Tripwires")
    for tw in red_team_result.active_tripwires:
        if tw.triggered:
            st.error(f"{tw.name}: {tw.value:.1f} ({tw.threshold}) - {tw.clinical_reason}")
        else:
            st.success(f"{tw.name}: {tw.value:.1f} (normal)")

    st.subheader("Recommended Actions")
    for action in decision.actions:
        st.write(f"- **[{action.action_type}]** {action.description} *(Priority {action.priority})*")

    with st.expander("Decision reasoning"):
        st.write(decision.reasoning)


def main() -> None:
    init_state()
    st.title("🛡️ QuantumSepsis Shield — Real-Time Demo")
    st.markdown("**Adversarially-Safe Quantum-Classical System for Early Sepsis Detection**")

    try:
        runtime = load_runtime(_version=2)
    except Exception as exc:
        st.error(f"Failed to initialize runtime: {exc}")
        st.stop()

    # ── Sidebar: Scenario ──────────────────────────────────────────
    st.sidebar.header("Patient Scenario")
    scenario = st.sidebar.selectbox(
        "Select patient",
        list(SCENARIOS.keys()),
        format_func=lambda key: SCENARIOS[key]["label"],
    )
    st.sidebar.markdown(SCENARIOS[scenario]["desc"])

    # ── Sidebar: Mode ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    demo_mode = st.sidebar.toggle("⚡ Demo Mode (instant results)", value=True)

    if demo_mode:
        mode = "Demo Instant"
        mode_help = "Results update instantly on every change."
    else:
        mode = st.sidebar.radio(
            "Cadence mode",
            ["Manual (click to run)", "Simulated 15-Min Cycle"],
            index=0,
        )
        mode_help = "Click button or use timer."

    if st.session_state.last_mode != mode:
        st.session_state.sim_running = False
        st.session_state.last_mode = mode

    st.sidebar.caption(mode_help)

    # ── Sidebar: Custom sliders ────────────────────────────────────
    custom_values = None
    if scenario == "custom":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Enter Vitals")
        custom_values = {
            "heart_rate": st.sidebar.slider("Heart Rate (bpm)", 30, 200, 90, key="cv_hr"),
            "map": st.sidebar.slider("MAP (mmHg)", 30, 140, 85, key="cv_map"),
            "temperature": st.sidebar.slider("Temperature (°C)", 33.0, 42.0, 37.0, 0.1, key="cv_temp"),
            "resp_rate": st.sidebar.slider("Resp Rate (br/min)", 5, 45, 16, key="cv_rr"),
            "spo2": st.sidebar.slider("SpO2 (%)", 70, 100, 98, key="cv_spo2"),
            "gcs_total": st.sidebar.slider("GCS Total", 3, 15, 15, key="cv_gcs"),
            "lactate": st.sidebar.slider("Lactate (mmol/L)", 0.5, 15.0, 1.2, 0.1, key="cv_lac"),
            "wbc": st.sidebar.slider("WBC (K/uL)", 0.5, 40.0, 8.0, 0.5, key="cv_wbc"),
            "creatinine": st.sidebar.slider("Creatinine (mg/dL)", 0.3, 8.0, 1.0, 0.1, key="cv_cr"),
            "platelets": st.sidebar.slider("Platelets (K/uL)", 10.0, 400.0, 200.0, 5.0, key="cv_plt"),
        }

    # ── Determine whether to run ───────────────────────────────────
    timeline_label = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_now = False
    sleep_seconds = None

    if mode == "Demo Instant":
        # Always run on every rerender (slider change, scenario change)
        run_now = True
    elif mode == "Manual (click to run)":
        run_now = st.sidebar.button("▶️ Run Prediction", type="primary", use_container_width=True)
    else:
        # Simulated 15-Min Cycle
        st.sidebar.markdown("---")
        speed_label = st.sidebar.selectbox(
            "Demo speed (1 cycle = 15 simulated minutes)",
            ["1 sec", "3 sec", "5 sec", "10 sec"],
            index=1,
        )
        speed_map = {"1 sec": 1, "3 sec": 3, "5 sec": 5, "10 sec": 10}
        sleep_seconds = speed_map[speed_label]

        c1, c2 = st.sidebar.columns(2)
        if c1.button("▶ Start"):
            st.session_state.sim_running = True
        if c2.button("⏸ Pause"):
            st.session_state.sim_running = False

        c3, c4 = st.sidebar.columns(2)
        if c3.button("Step +15 min"):
            st.session_state.sim_time += timedelta(minutes=15)
            st.session_state.cycle_id += 1
            run_now = True
        if c4.button("🔄 Reset"):
            st.session_state.sim_time = datetime.now().replace(second=0, microsecond=0)
            st.session_state.cycle_id = 0
            st.session_state.sim_running = False

        if st.session_state.sim_running:
            st.session_state.sim_time += timedelta(minutes=15)
            st.session_state.cycle_id += 1
            run_now = True

        timeline_label = st.session_state.sim_time.strftime("%Y-%m-%d %H:%M")

    # ── Execute prediction ─────────────────────────────────────────
    if run_now:
        if scenario == "custom" and custom_values is not None:
            window = generate_custom_window(custom_values)
        else:
            seed = (
                st.session_state.cycle_id
                if mode == "Simulated 15-Min Cycle"
                else int(time.time() // 60)
            )
            window = generate_patient_window(scenario, seed)
        run_prediction(runtime, scenario, window, timeline_label)

    # ── Render ─────────────────────────────────────────────────────
    result = st.session_state.latest_result
    window = st.session_state.latest_window

    if result is None or window is None:
        st.info("👉 Choose a scenario (or enter custom vitals) to start the demo.")
    else:
        render_dashboard(result, window, scenario, mode, timeline_label)

    st.markdown("---")
    st.caption("🛡️ QuantumSepsis Shield © 2026 — Yash Gautam · Atul Kumar Mishra · Tanishk Viraj Bhanage")

    if mode == "Simulated 15-Min Cycle" and st.session_state.sim_running and sleep_seconds is not None:
        time.sleep(sleep_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
