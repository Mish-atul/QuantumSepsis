# QuantumSepsis Shield — Phase 3 Tasks (Publication-Ready)

> **Status**: Phase 1+2 Complete | Phase 3 In Progress  
> **Deadline**: GPU server access expires in ~2 days  
> **Goal**: Publication-grade results + real-time demo

---

## 📋 Quick Context

**Project**: QuantumSepsis Shield — early sepsis detection on MIMIC-IV v3.1  
**Repo**: https://github.com/Mish-atul/QuantumSepsis  
**GPU Server**: `ssh csegpuserver@172.16.18.2` (password: `Redhat#84@`)  
**Working Directory**: `~/QuantumSepsis/`  
**Critical Rule**: Always `export PYTHONPATH=.` before running Python scripts

### Phase 1+2 Results Summary

| Model | Test AUROC | Test AUPRC | Status |
|-------|-----------|------------|--------|
| SOFA Baseline | 0.5869 | 0.0159 | ✅ Clinical baseline |
| **XGBoost** | **0.8038** | **0.0576** | ✅ **Best model** |
| LSTM | 0.7891 | 0.0519 | ✅ Deep learning |
| RBF Quantum Kernel | 0.7879 | 0.0520 | ✅ Validated |
| Qiskit Quantum | ❌ Crashed | ❌ Crashed | ❌ Computationally infeasible |

**What's Done**:
- ✅ Full MIMIC-IV pipeline (94,458 ICU stays, 4.09M windows)
- ✅ LSTM trained and embeddings extracted
- ✅ Quantum kernel bug fixed and validated via RBF
- ✅ All Phase 2 integration scripts written
- ✅ All critical data backed up

**What's Needed**:
- 🔧 Fix Red Team denormalization bug
- 🚀 Run Phase 2 scripts on real data
- 📈 Tune LSTM to beat XGBoost
- 📊 Compute stay-level metrics
- 🖥️ Build real-time Streamlit dashboard

---

## 🎯 TASK 1 — FIX RED TEAM BUG (30 min) [CRITICAL]

### Problem
Red Team Agent reported 100/100 windows as CRITICAL. This is a **bug**, not a feature. The clinical thresholds (MAP < 70 mmHg, Temp > 38.3°C) are designed for **raw clinical values**, but the pipeline feeds **z-normalized data**. A MAP z-score of -0.5 doesn't mean hypotension.

### Root Cause
The `RedTeamAgent` was initialized with `use_normalized=False` (default), so it applied raw clinical thresholds to normalized data.

### Fix & Test

```bash
cd ~/QuantumSepsis && export PYTHONPATH=.

python3 << 'PYEOF'
import json, numpy as np, h5py
from src.agents.red_team import RedTeamAgent
from src.config import get_default_config

# Load normalization stats
norm_stats = json.load(open('data/processed/normalization_stats.json'))

# Initialize Red Team with denormalization enabled
agent = RedTeamAgent(
    config=get_default_config().red_team,
    use_normalized=True,
    norm_stats=norm_stats
)

# Test on 200 windows
with h5py.File('data/processed/features.h5', 'r') as f:
    X = f['X_test'][:200]
    y = f['y_test'][:200]

counts = {}
for w in X:
    result = agent.evaluate(w)
    counts[result.override_level] = counts.get(result.override_level, 0) + 1

print('Red Team (with denormalization):')
for level in ['WATCH', 'AMBER', 'CRITICAL']:
    pct = counts.get(level, 0) / 2.0
    print(f'  {level}: {counts.get(level, 0)}/200 ({pct:.0f}%)')
print(f'\nLabels: {int(y.sum())} sepsis, {int((1-y).sum())} non-sepsis')
PYEOF
```

### Expected Output
- **WATCH**: ~60-70% (120-140 windows)
- **AMBER**: ~20-25% (40-50 windows)
- **CRITICAL**: ~5-15% (10-30 windows)

### If Still 100% CRITICAL
Check that `normalization_stats.json` has the correct structure:
```json
{
  "train_mean": {
    "heart_rate": 85.2,
    "sbp": 120.5,
    ...
  },
  "train_std": {
    "heart_rate": 15.3,
    "sbp": 18.7,
    ...
  }
}
```

### Deliverable
Report the actual distribution (% WATCH/AMBER/CRITICAL) after the fix.

---

## 🚀 TASK 2 — RUN PHASE 2 SCRIPTS ON REAL DATA (2-3 hrs)

These scripts exist in the repo but haven't been run on real data yet. Run them in order:

### 2.1 Conformal Calibration

```bash
cd ~/QuantumSepsis && export PYTHONPATH=.
python3 scripts/run_conformal_calibration.py
```

**What it does**: Calibrates the conformal predictor on LSTM validation scores, verifies ≥90% coverage on test set.

**Expected outputs**:
- `data/processed/conformal_calibration.json`
- `data/processed/conformal_test_intervals.npz`

**Report**: Coverage achieved (target ≥ 90%)

### 2.2 End-to-End Validation

```bash
python3 scripts/run_e2e_validation.py
```

**What it does**: Wires LSTM + Conformal + RedTeam (with denormalization!) + Orchestrator on test set.

**Expected outputs**:
- `data/processed/e2e_validation_results.json`
- `data/processed/e2e_decisions.npz`

**Report**: Alert distribution (% WATCH/AMBER/CRITICAL/FAST-TRACK), sensitivity, specificity

### 2.3 Outcome Learning Simulation

```bash
python3 scripts/run_outcome_learning_simulation.py
```

**What it does**: Feeds decisions into OutcomeLearningAgent, tracks FN/near-misses, simulates adaptive threshold updates.

**Expected outputs**:
- `data/processed/outcome_learning_results.json`
- `data/processed/near_miss_weights.json`

**Report**: FN rate reduction over rounds, near-miss count

### 2.4 Class Imbalance Analysis

```bash
python3 scripts/analyze_class_imbalance.py
```

**What it does**: Investigates why AUPRC=0.05, analyzes focal gamma sensitivity.

**Expected output**:
- `data/processed/class_imbalance_analysis.json`

**Report**: Window-level positive rate, AUPRC baseline, focal loss impact

### Troubleshooting
If any script fails with missing arguments:
```bash
python3 scripts/[script_name].py --help
```
Then pass the correct paths (likely `--data data/processed/features.h5` and `--checkpoint checkpoints/lstm_best.pt`).

---

## 📈 TASK 3 — LSTM TUNING TO BEAT XGBOOST (3-4 hrs on A100)

### Goal
Beat XGBoost's 0.8038 test AUROC with a tuned LSTM.

### Run Command

```bash
screen -S tuning
cd ~/QuantumSepsis && export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lstm_tuning.py --exp exp5_combined
# Ctrl+A, D to detach
```

### What It Does
Runs 5 tuning experiments:
1. **exp1_deeper**: 3 LSTM layers instead of 2
2. **exp2_wider**: hidden_dim=256 instead of 128
3. **exp3_higher_focal**: focal_gamma=3.0 instead of 2.0
4. **exp4_lower_lr**: learning_rate=0.0005 instead of 0.001
5. **exp5_combined**: Best hyperparameters from exp1-4

### Expected Output
- `data/processed/tuning_results.json` (per experiment)
- Best model checkpoint: `checkpoints/lstm_tuned_best.pt`

### Monitoring
```bash
screen -r tuning  # Reattach to see progress
# Or check logs:
tail -f logs/lstm_tuning_*.log
```

### Deliverable
Report the best tuned LSTM test AUROC. Target: ≥ 0.8038 (beat XGBoost).

---

## 📊 TASK 4 — STAY-LEVEL METRICS (1-2 hrs)

### Problem
Current metrics are **window-level** (4.09M windows). Published sepsis papers report **stay-level** metrics (94K stays). This is the standard methodology.

### Approach
For each `stay_id`, aggregate all its windows by taking `max(risk_score)`. If any window predicts sepsis, the stay is flagged.

### Implementation

Create `scripts/compute_stay_level_metrics.py`:

```python
"""
Aggregate window-level predictions to stay-level and compute AUROC.
"""
import numpy as np
import h5py
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

def main():
    # Load window predictions (from LSTM or e2e validation)
    with h5py.File('data/processed/features.h5', 'r') as f:
        y_test = f['y_test'][:]
    
    # Load risk scores (from e2e_decisions.npz or lstm inference)
    decisions = np.load('data/processed/e2e_decisions.npz')
    risk_scores = decisions['risk_scores']
    
    # Load stay_id mapping
    # Option 1: If features.h5 has stay_ids
    # with h5py.File('data/processed/features.h5', 'r') as f:
    #     stay_ids = f['stay_ids_test'][:]
    
    # Option 2: Reconstruct from windowing metadata
    # Load data/processed/test_features.parquet and match by index
    import pandas as pd
    test_df = pd.read_parquet('data/processed/test_features.parquet')
    
    # Group by stay_id
    stay_scores = defaultdict(list)
    stay_labels = {}
    
    for i, (score, label) in enumerate(zip(risk_scores, y_test)):
        stay_id = test_df.iloc[i]['stay_id']  # Adjust based on actual structure
        stay_scores[stay_id].append(score)
        stay_labels[stay_id] = max(stay_labels.get(stay_id, 0), label)
    
    # Aggregate: max risk score per stay
    stay_ids = list(stay_scores.keys())
    y_stay = np.array([stay_labels[sid] for sid in stay_ids])
    scores_stay = np.array([max(stay_scores[sid]) for sid in stay_ids])
    
    # Compute stay-level metrics
    auroc = roc_auc_score(y_stay, scores_stay)
    auprc = average_precision_score(y_stay, scores_stay)
    
    results = {
        'n_stays': len(stay_ids),
        'n_sepsis_stays': int(y_stay.sum()),
        'stay_level_auroc': float(auroc),
        'stay_level_auprc': float(auprc),
    }
    
    print(json.dumps(results, indent=2))
    
    with open('data/processed/stay_level_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
```

### Run

```bash
cd ~/QuantumSepsis && export PYTHONPATH=.
python3 scripts/compute_stay_level_metrics.py
```

### Expected Output
- **Stay-level AUROC**: 0.85-0.88 (higher than window-level because aggregation reduces noise)
- **Stay-level AUPRC**: 0.15-0.25 (higher than window-level 0.05)

### Note
The exact implementation depends on how `stay_id` is stored. Check:
1. `features.h5` for a `stay_ids_test` dataset
2. `data/processed/test_features.parquet` for stay_id column
3. Windowing logs for stay_id → window_index mapping

### Deliverable
Report stay-level AUROC and AUPRC.

---

## 🖥️ TASK 5 — STREAMLIT REAL-TIME DASHBOARD (3-4 hrs, can do locally)

### Goal
Create a real-time demo dashboard for the final presentation. This simulates hardware integration (ICU monitor → AI system → clinical alert).

### Implementation

Create `scripts/realtime_demo.py`:

```python
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
    conformal.load('data/processed/conformal_calibration.json')
    
    # Red Team
    norm_stats = json.load(open('data/processed/normalization_stats.json'))
    red_team = RedTeamAgent(config.red_team, use_normalized=True, norm_stats=norm_stats)
    
    # Orchestrator
    orchestrator = ConfidenceGatedOrchestrator(config.orchestrator)
    
    return lstm, conformal, red_team, orchestrator

lstm, conformal, red_team, orchestrator = load_models()

# Simulate 3 patient scenarios
def generate_patient_window(scenario, hour):
    """Generate synthetic 6-hour window for demo."""
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
    window = np.stack([hr, map_val, map_val, map_val, temp, rr, spo2, gcs,
                       np.ones(6)*1.5, np.ones(6)*10, np.ones(6)*1.0, np.ones(6)*200], axis=1)
    return window

# Main dashboard
st.title("🛡️ QuantumSepsis Shield — Real-Time Demo")
st.markdown("**Adversarially-Safe Quantum-Classical System for Early Sepsis Detection**")

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

# Normalize (simplified - use actual normalization stats)
window_norm = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-8)

# Run inference
with torch.no_grad():
    window_tensor = torch.FloatTensor(window_norm).unsqueeze(0)
    outputs = lstm(window_tensor)
    risk_score = outputs['risk_score'].item()

# Conformal prediction
lower, upper = conformal.predict_interval(risk_score)
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
import plotly.graph_objects as go

hours = list(range(-5, 1))
df = pd.DataFrame({
    'Hour': hours,
    'Heart Rate': window[:, 0],
    'MAP': window[:, 3],
    'Temperature': window[:, 4],
    'Resp Rate': window[:, 5],
    'SpO2': window[:, 6],
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Hour'], y=df['Heart Rate'], name='HR', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['Hour'], y=df['MAP'], name='MAP', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['Hour'], y=df['SpO2'], name='SpO2', line=dict(color='green')))
fig.update_layout(height=300, xaxis_title="Hours Ago", yaxis_title="Value")
st.plotly_chart(fig, use_container_width=True)

# Tripwire panel
st.subheader("🚨 Clinical Tripwires")
for tw in red_team_result.active_tripwires:
    if tw.triggered:
        st.error(f"**{tw.name}**: {tw.value:.1f} ({tw.threshold}) — {tw.clinical_reason}")

# Clinical actions
st.subheader("📋 Recommended Actions")
for action in decision.actions:
    st.write(f"- {action}")

# Reasoning
with st.expander("🧠 Decision Reasoning"):
    st.write(decision.reasoning)
```

### Run Locally

```bash
# On your local machine (not server)
cd ~/QuantumSepsis
pip install streamlit plotly
streamlit run scripts/realtime_demo.py
```

### Expected Output
- Dashboard opens in browser at `http://localhost:8501`
- Shows 3 patient scenarios with live vital signs
- Updates every 60 seconds if auto-refresh enabled
- Displays risk gauge, alert level, tripwires, and clinical actions

### Deliverable
- Screenshot of the dashboard showing all 3 scenarios
- Or confirmation that it runs successfully

---

## 📤 TASK 6 — COMMIT AND REPORT

After **each task**, commit and push to GitHub:

```bash
cd ~/QuantumSepsis
git add -A
git commit -m "Phase 3: [task description]"
git push origin main
```

### Final Report Required

Create a summary with these exact metrics:

#### 1. Red Team Distribution (after denormalization fix)
- WATCH: X/200 (X%)
- AMBER: X/200 (X%)
- CRITICAL: X/200 (X%)

#### 2. Conformal Prediction
- Coverage achieved: X% (target ≥ 90%)
- Average interval width: X.XX

#### 3. Outcome Learning
- Initial FN rate: X%
- Final FN rate after 5 rounds: X%
- Near-miss count: X

#### 4. LSTM Tuning
- Best experiment: expX_[name]
- Tuned LSTM test AUROC: X.XXXX
- Beat XGBoost? [Yes/No]

#### 5. Stay-Level Metrics
- Stay-level AUROC: X.XXXX (expected 0.85-0.88)
- Stay-level AUPRC: X.XXXX

#### 6. Dashboard
- Status: [Running/Screenshot attached]
- Scenarios tested: [All 3/Partial]

---

## ⚠️ Important Notes

### GPU Server Access
- **Expires in ~2 days** — prioritize Tasks 1-4 on the server
- Task 5 (dashboard) can be done locally
- Always use `screen` for long-running jobs
- Check `nvidia-smi` before using GPU

### Data Paths
All scripts expect:
- `data/processed/features.h5` — windowed data
- `checkpoints/lstm_best.pt` — trained LSTM
- `data/processed/normalization_stats.json` — for Red Team denormalization

### Troubleshooting
If a script fails:
1. Check `--help` for required arguments
2. Verify file paths exist
3. Check logs in `logs/` directory
4. Ensure `export PYTHONPATH=.` is set

### Success Criteria
- ✅ Red Team shows realistic distribution (not 100% CRITICAL)
- ✅ Conformal coverage ≥ 90%
- ✅ Tuned LSTM AUROC ≥ 0.8038 (beats XGBoost)
- ✅ Stay-level AUROC ≥ 0.85
- ✅ Dashboard runs and displays all scenarios

---

## 📚 Additional Resources

- **Full project documentation**: `agents.md` (complete knowledge base)
- **Phase 1+2 results**: `PROGRESS_REPORT.md`
- **Architecture details**: `files/architecture.md`
- **Novelty claims**: `files/novelty.md`

---

**Good luck! This is the final push to publication-grade results. 🚀**
