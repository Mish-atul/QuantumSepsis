# QuantumSepsis Shield — Model Improvement V2

> **Target:** Improve AUROC from 0.79 → 0.85+ (window), 0.86 → 0.92+ (stay)  
> **New novelty:** Conformal-Gated Quantum-Classical Ensemble  
> **Timeline:** 1 week

---

## Problem Statement

Current V1 results are competitive with PhysioNet challenge winners but below
state-of-the-art deep learning papers that achieve AUROC 0.88–0.93:

| Model (V1) | Window AUROC | Stay AUROC |
|-------------|-------------|------------|
| SOFA Baseline | 0.5869 | — |
| LSTM V1 | 0.7891 | 0.8618 |
| XGBoost | 0.8038 | — |
| Qiskit Quantum | 0.7598 | — |

**Root causes:**
1. Only 12 raw features (top papers use 30–40)
2. Vanilla BiLSTM (no multi-scale temporal modeling)
3. No quantum-classical fusion strategy

---

## Three Improvements

### 1. Feature Engineering: 12 → 33 Features

**File:** `src/data/feature_engineering_v2.py`

Adds 21 derived features computed at each time step within the 6-hour window.
All features are causal (no future data leakage) and computable at inference time.

| Category | Features | Count |
|----------|----------|-------|
| **Delta (rate of change)** | ΔHR, ΔSBP, ΔMAP, Δlactate, ΔRR, Δtemp | 6 |
| **Clinical indices** | Shock Index (HR/SBP), Modified SI (HR/MAP), Pulse Pressure (SBP-DBP), SpO2/HR ratio | 4 |
| **Rolling statistics** | HR std, MAP std, lactate max, temp range, RR std (3hr rolling) | 5 |
| **Trend slopes** | HR slope, MAP slope, lactate slope (linear regression over window) | 3 |
| **Interactions** | HR×temp, lactate/MAP, WBC/platelets | 3 |
| **Total** | | **21** |

**Expected AUROC boost: +0.04 to +0.06**

### 2. Enhanced Model: SepsisLSTMv2

**File:** `src/models/lstm_v2.py`

```
Input: (batch, 6, 33)
  → MultiScaleConv1D (kernels: 2, 3, 5) → 96 channels
  → ChannelAttention (SE-block, learns feature importance)
  → Residual: conv_out + projected_raw → 96 dim
  → LayerNorm
  → BiLSTM (128 hidden × 2 directions = 256) × 2 layers
  → MultiHeadTemporalAttention (4 heads × 64 attn_dim)
  → FC: 256 → 128 → GELU → Dropout(0.3)
  → FC: 128 → 16 → Tanh  (quantum embedding, [-1,1])
  → FC: 16 → 1 → Sigmoid (risk score)
```

**Key improvements over V1:**
- Multi-scale convolutions capture 2h/3h/5h temporal patterns
- Channel attention learns which features matter most
- Multi-head (4 heads) temporal attention vs single-head
- Residual connections improve gradient flow
- GELU activation (smoother than ReLU)
- ~800K parameters (vs ~420K in V1)

**Expected AUROC boost: +0.02 to +0.04**

### 3. Conformal-Gated Quantum-Classical Ensemble (NEW NOVELTY)

**File:** `src/models/ensemble.py`

Uses conformal prediction interval width as a dynamic gating signal:
- When LSTM is **confident** (narrow interval) → trust LSTM more
- When LSTM is **uncertain** (wide interval) → weight quantum kernel more

```python
confidence = 1.0 - conformal_width
gate = sigmoid(β × (confidence - τ))
ensemble = gate × lstm_score + (1 - gate) × quantum_score
```

Parameters β and τ are calibrated on the validation set via grid search.

**Why this is novel:** No prior work uses conformal interval width as a dynamic
fusion signal between quantum and classical models. This creates a coherent
system where Novelty 1 (QCCP) directly improves the prediction pipeline.

---

## Training Pipeline

**Script:** `scripts/train_v2.py`

```bash
# On GPU server:
screen -S train_v2
cd ~/QuantumSepsis && export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_v2.py \
    --data data/processed/features.h5 \
    --epochs 100 \
    --batch-size 256 \
    --patience 15

# Local smoke test:
python3 scripts/train_v2.py --synthetic
```

**Pipeline flow:**
```
features.h5 (N, 6, 12)
    ↓ feature_engineering_v2.py
enriched (N, 6, 33)
    ↓ normalize derived features
normalized (N, 6, 33)
    ↓ train_v2.py
lstm_v2_best.pt + lstm_v2_embeddings.npz
    ↓ quantum_kernel.py (same as before, on new embeddings)
quantum_results_v2.json
    ↓ ensemble.py (calibrate on val set)
ensemble_results_v2.json
```

## Output Files

| File | Description |
|------|-------------|
| `checkpoints/lstm_v2_best.pt` | Best V2 model checkpoint |
| `data/processed/lstm_v2_embeddings.npz` | 16-dim embeddings for quantum kernel |
| `data/processed/v2_normalization_stats.json` | Normalization stats for derived features |
| `data/processed/v2_training_results.json` | Training history + test metrics |
| `data/processed/quantum_results_v2.json` | Quantum kernel on V2 embeddings |
| `data/processed/ensemble_results_v2.json` | Ensemble calibration results |

## Compatibility

- **Dashboard:** Works — raw vitals → feature enrichment → model inference
- **Red Team:** Unchanged — operates on raw vitals independently
- **Orchestrator:** Unchanged — takes risk_score + conformal interval
- **Conformal:** Re-calibrate on V2 val scores (same API)
- **Quantum Kernel:** Re-run on V2 embeddings (same API, new npz file)

## Target Results

| Model | Current | Target |
|-------|---------|--------|
| LSTM V2 (window) | — | **0.84–0.88** |
| LSTM V2 (stay) | — | **0.90–0.95** |
| Ensemble (window) | — | **0.85–0.90** |
| Ensemble (stay) | — | **0.92–0.96** |
