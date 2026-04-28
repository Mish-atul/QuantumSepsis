# QuantumSepsis Shield — Complete Project Knowledge Base

> **Project:** Adversarially-Safe Quantum-Classical System for Early Sepsis Detection  
> **Team:** Yash Gautam · Atul Kumar Mishra · Tanishk Viraj Bhanage  
> **Last Updated:** April 29, 2026

---

## 1. Project Overview

Sepsis kills **11 million people per year** globally. Early detection saves lives — every hour of delay increases mortality by ~7%. QuantumSepsis Shield is a 5-layer AI pipeline that detects sepsis **3–4 hours before clinical onset** using:

- **Classical deep learning** (BiLSTM with temporal attention)
- **Quantum kernel methods** (Qiskit ZZFeatureMap + SVM)
- **Conformal prediction** (statistically guaranteed uncertainty intervals)
- **Adversarial safety agents** (non-overridable clinical tripwires)
- **Adaptive outcome learning** (per-ICU-unit threshold tuning)

The system outputs one of four clinical alert levels: **WATCH / AMBER / CRITICAL / FAST-TRACK**.

---

## 2. Dataset — MIMIC-IV v3.1

### Source
- **Name:** Medical Information Mart for Intensive Care IV
- **Version:** 3.1 (latest, 2023)
- **Source:** PhysioNet / Beth Israel Deaconess Medical Center
- **Access:** Requires PhysioNet credentialed account + DUA

### Scale
| Table | Rows | Usage |
|---|---|---|
| patients | 364,627 | Demographics |
| admissions | 546,028 | Hospitalization records |
| icustays | 94,458 | ICU boundaries (our cohort base) |
| prescriptions | ~17M | Antibiotic orders → suspected infection |
| microbiologyevents | ~600K | Blood cultures → suspected infection |
| labevents | **~124M** | Lactate, WBC, Creatinine, Platelets |
| chartevents | **~330M** | HR, BP, Temp, SpO2, RR, GCS |

### Cohort Definition (Sepsis-3 Criteria)
A patient is labeled **sepsis-positive** if ALL of the following are true:
1. **Suspected infection** = antibiotic order + blood/body culture within ±24 hours
2. **Organ dysfunction** = SOFA score increase ≥ 2 from first-24h baseline
3. **Onset time** = `max(suspected_infection_time, sofa_increase_time)`

### Final Cohort (on GPU server)
| Metric | Value |
|---|---|
| Total ICU stays | 94,458 |
| Sepsis-positive | 12,972 (13.7%) |
| Sepsis-negative | 81,486 (86.3%) |

### Train/Val/Test Split
- **Temporal split** by `anchor_year_group` (prevents data leakage)
- **Train:** 2008–2019 (4 year-groups)
- **Test:** 2020–2022
- **Val:** 15% random stratified sample from training set

### 12 Input Features
| # | Feature | Source | Item IDs | Unit |
|---|---|---|---|---|
| 0 | heart_rate | chartevents | 211, 220045 | bpm |
| 1 | sbp | chartevents | 51, 442, 455, 6701, 220179, 220050 | mmHg |
| 2 | dbp | chartevents | 8368, 8440, 8441, 8555, 220180, 220051 | mmHg |
| 3 | map | chartevents | 52, 456, 6702, 220052, 220181 | mmHg |
| 4 | temperature | chartevents | 223762, 226329 | °C |
| 5 | resp_rate | chartevents | 615, 618, 220210, 224690 | br/min |
| 6 | spo2 | chartevents | 646, 220277 | % |
| 7 | gcs_total | chartevents | 198, 226755, 227013 | score |
| 8 | lactate | labevents | 50813 | mmol/L |
| 9 | wbc | labevents | 51301 | K/uL |
| 10 | creatinine | labevents | 50912 | mg/dL |
| 11 | platelets | labevents | 51265 | K/uL |

---

## 3. Data Pipeline (src/data/)

### Stage 1 — Cohort Extraction

**Files:** `cohort_extraction.py` (broken), `cohort_extraction_optimized.py` (✅ use this)

**What it does:**
- Loads `icustays`, `admissions`, `patients`, `prescriptions`, `microbiologyevents`
- Computes SOFA score components from `labevents` and `chartevents`
- Applies Sepsis-3 criteria to label each ICU stay
- Outputs `data/processed/cohort.csv` with columns: `stay_id, hadm_id, subject_id, intime, outtime, sepsis_label, sepsis_onset_time, anchor_year_group`

**Critical Bug Fixed:** Original version loaded all 120M labevents + 330M chartevents into RAM → OOM kill. Fixed version uses **chunked CSV reading** (500K rows/chunk) with inline filtering:
```python
def load_table_chunked(data_dir, table_name, module, usecols,
                       filter_col, filter_values, chunksize=500_000):
    reader = pd.read_csv(filepath, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        filtered = chunk[chunk[filter_col].isin(filter_values)]
        chunks.append(filtered)
    return pd.concat(chunks)
# Memory: ~30 GB → ~2-3 GB
```

**Run command:**
```bash
python3 -m src.data.cohort_extraction_optimized \
    --data-dir data/raw/physionet.org/files/mimiciv/3.1
```

---

### Stage 2 — Feature Extraction

**File:** `feature_extraction.py`

**What it does:**
- For each ICU stay, bins time into 1-hour intervals
- Loads chartevents filtered to the 35 vital-sign item IDs
- Loads labevents filtered to the 4 lab item IDs
- Aggregates (mean) per hour per feature
- Outputs `data/processed/hourly_features.parquet` (~56 MB)

**Note:** This file also has the OOM bug on `_load_vitals()` and `_load_labs()` — must use `load_table_chunked` from `cohort_extraction_optimized.py`.

**Run command:**
```bash
python3 -m src.data.feature_extraction \
    --cohort data/processed/cohort.csv \
    --data-dir data/raw/physionet.org/files/mimiciv/3.1
```

---

### Stage 3 — Preprocessing

**File:** `preprocessing.py`

**What it does:**
1. **Forward-fill** missing values (limit: 2 hours)
2. **Median imputation** for remaining NaNs (per-feature train-set medians)
3. **Z-score normalization** (train-set mean/std, applied to val/test)
4. **Temporal train/val/test split** by `anchor_year_group`
5. Saves `normalization_stats.json` for the Red Team Agent's denormalization

**Outputs:**
- `data/processed/train_features.parquet`
- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`
- `data/processed/normalization_stats.json`

---

### Stage 4 — Windowing

**File:** `windowing.py`  
**Runner:** `scripts/run_windowing_real.py`

**What it does:**
- Generates **6-hour sliding windows** with 1-hour stride
- Each window: shape `(6, 12)` = 6 time steps × 12 features
- Label: 1 if sepsis onset occurs within next 4 hours, 0 otherwise
- Stores everything in HDF5 format

**Output:** `data/processed/features.h5`
```
features.h5
├── X_train  — (N_train, 6, 12)
├── y_train  — (N_train,)
├── X_val    — (N_val, 6, 12)
├── y_val    — (N_val,)
├── X_test   — (N_test, 6, 12)
└── y_test   — (N_test,)
```

**Scale:** ~4.09M training windows (vast majority negative → high class imbalance)

---

### Stage 5 — Dataset / DataLoader

**File:** `dataset.py`

PyTorch `Dataset` and `DataLoader` wrappers over `features.h5`. Handles batch loading of `(X, y)` tensors for training.

---

## 4. Models (src/models/)

### 4.1 SepsisLSTM — `lstm.py`

**Architecture:**
```
Input: (batch, 6, 12)
  → LayerNorm([6, 12])
  → BiLSTM(input=12, hidden=128, layers=2, dropout=0.3)  → (batch, 6, 256)
  → TemporalAttention(256, attn_dim=64)                   → (batch, 256)
  → FC(256 → 64, ReLU, Dropout(0.3))
  → FC(64 → 16, Tanh)                    ← 16-dim embedding (bounded [-1,1])
  → FC(16 → 1, Sigmoid)                  ← risk_score ∈ [0,1]
```

**Key design decisions:**
- **Bidirectional LSTM:** Reads the 6-hour window both forward and backward, capturing both early warning signs and deterioration trends
- **TemporalAttention:** Learns which hours in the window are most informative for each patient. Uses a 2-layer MLP: `Linear(256→64) → Tanh → Linear(64→1)` then Softmax over time steps
- **Tanh embedding:** Bounds the 16-dim embedding to [-1, 1], which is required for angle-encoding in the quantum feature map
- **Total parameters:** ~420K

**Outputs (dict):**
- `logits` — raw pre-sigmoid output
- `risk_score` — sigmoid probability ∈ [0, 1]
- `embedding` — 16-dim latent vector (quantum kernel interface)
- `attention_weights` — 6-dim attention distribution (interpretability)

**Key method:** `extract_embeddings(x)` → returns only the 16-dim embedding for quantum kernel use

---

### 4.2 AsymmetricFocalLoss — `losses.py`

**Motivation:** Sepsis has ~13.7% prevalence in cohort but much lower per-window positive rate (~few %). False negatives (missing sepsis) are far more dangerous than false positives (unnecessary workup).

**Formula:**
```
L = -α_pos × (1-p)^γ × log(p)   [for positives, FN penalty]
  + -α_neg × p^γ × log(1-p)     [for negatives, FP penalty]
```

**Config values:**
- `α_pos = 0.9` (high weight on positive class)
- `α_neg = 0.1` (low weight on negative class)
- `γ = 2.0` (focusing parameter — down-weights easy examples)
- **FN:FP penalty ratio ≈ 9:1**

---

### 4.3 QuantumKernelSepsis — `quantum_kernel.py`

**Purpose:** Phase 2 classifier that operates on LSTM embeddings using quantum kernel methods.

**Full workflow:**
```
lstm_embeddings.npz (N, 16)
  → Balanced subsample (2000 samples: 1000 pos + 1000 neg)
  → PCA: 16-dim → 8-dim (99.25% explained variance)
  → Quantum Kernel Matrix K[i,j] = |⟨φ(x_i)|φ(x_j)⟩|²
  → SVM with precomputed kernel (C=0.1, class_weight={0:1, 1:10})
  → Inference: support-vector-only (avoids recomputing full kernel)
```

**Quantum circuit:** ZZFeatureMap
- **Qubits:** 8 (matches PCA output dimension)
- **Repetitions:** 2
- **Entanglement:** Linear (qubit i entangles with qubit i+1)
- **Backend:** Qiskit AerSimulator (1024 shots per circuit)
- **Kernel value:** Quantum fidelity K(x,y) = |⟨φ(x)|φ(y)⟩|² ∈ [0, 1]

**Fallback:** If Qiskit not installed → RBF kernel (classical SVM). RBF results: AUROC 0.7879.

**Why subsample?** A full 4M×4M kernel matrix is impossible. 2000×2000 is tractable. Inference uses only support vectors (typically ~87% of 2000 = ~1734 vectors).

**Key methods:**
- `balanced_subsample(X, y, max_samples)` — creates balanced train subset
- `fit_pca(embeddings)` — fits PCA 16→8
- `setup_qiskit_kernel()` — initializes FidelityQuantumKernel
- `fit(X_train, y_train)` — trains QSVM
- `predict_scores(X)` — inference via support vectors only
- `get_centroids(X_train, y_train, n=5)` — computes sepsis centroids for QCCP

---

### 4.4 Conformal Prediction — `conformal.py`

Two classes:

**`ConformalSepsisPredictor` (standard split conformal):**
- Input: LSTM risk scores + true labels on calibration set
- Nonconformity score: `s_i = |y_i - f(x_i)|`
- Threshold: `q_α = quantile(scores, (1-α)(1+1/n))` at α=0.10 → 90% coverage
- Prediction: `[risk - q_α, risk + q_α]` clipped to [0, 1]
- Escalation rule: interval width > 0.4 → upgrade alert level

**`QuantumCalibratedConformal` (QCCP — Novelty 1):**
- Uses quantum kernel distance as nonconformity score instead of label residual
- `s(x) = 1 - max_j K(x, c_j)` where c_j are sepsis centroids in Hilbert space
- Tighter intervals because quantum kernel captures non-linear structure
- Requires: fitted quantum kernel + computed centroids
- **Status:** Code complete, not yet run on real data

**Coverage guarantee (mathematical):**
```
P(y_true ∈ [lower, upper]) ≥ 1 - α = 90%
```
This is distribution-free — valid regardless of model architecture.

---

## 5. Training Pipeline (src/training/)

### `train_lstm.py`

**Full training loop:**
1. Loads `features.h5` → `SepsisDataset` → `DataLoader`
2. Initializes `SepsisLSTM` + `AsymmetricFocalLoss`
3. Optimizer: **AdamW** (lr=0.001, weight_decay=0.0001)
4. Scheduler: **Cosine Annealing** (T_max=50 epochs)
5. **Early stopping:** patience=10 epochs on `val_auroc`
6. **Gradient clipping:** norm=1.0
7. **W&B logging:** project=`quantumsepsis-shield`
8. Saves best checkpoint to `checkpoints/lstm_best.pt`
9. After training: extracts 16-dim embeddings for all splits → `lstm_embeddings.npz`

**Real results (on A100):**
| Split | AUROC | AUPRC |
|---|---|---|
| Val | 0.7601 | — |
| Test | 0.7891 | 0.0519 |

**Run command:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm \
    --data data/processed/features.h5
```

---

## 6. Safety Agents (src/agents/)

### 6.1 RedTeamAgent — `red_team.py`

**Design principle:** Completely independent of the ML pipeline. Cannot be suppressed or overridden by model output. Even if quantum kernel says risk = 0.0, two simultaneous tripwires force a CRITICAL alert.

**5 Clinical Tripwires:**

| ID | Name | Condition | Clinical Reason |
|---|---|---|---|
| TW-TEMP | Temperature | < 36°C OR > 38.3°C | Hypothermia or fever (SIRS criterion) |
| TW-HR | Heart Rate | > 90 bpm AND upward trend > 5 bpm/hr | Tachycardia with deterioration |
| TW-RR | Respiratory Rate | > 20 breaths/min | Tachypnea (SIRS criterion) |
| TW-MAP | Mean Arterial Pressure | < 70 mmHg | Hypotension (Sepsis-3 cardiovascular) |
| TW-MENTAL | GCS Total | < 14 | Altered mental status |

**Escalation logic:**
```
≥ 2 tripwires active → CRITICAL (NON-OVERRIDABLE)
1 tripwire active   → AMBER
0 tripwires active  → WATCH
```

**Input:** `vitals_window` — shape `(T, 12)`, T time steps  
**Output:** `RedTeamAssessment` — `{triggered, active_tripwires, override_level, n_active, details}`

**Trend computation:** Linear regression slope on heart rate over the 6-hour window (bpm/hr). A patient with HR=92 but declining trend does NOT trigger TW-HR.

**Denormalization support:** If vitals are z-normalized (as they come from the preprocessing pipeline), the agent can denormalize using `normalization_stats.json` before applying thresholds.

---

### 6.2 ConfidenceGatedOrchestrator — `orchestrator.py`

**Purpose:** Fuses all signals into a single clinical decision.

**Inputs:**
- `risk_score` — from LSTM (or quantum kernel when available) ∈ [0, 1]
- `conformal_lower`, `conformal_upper` — prediction interval bounds
- `red_team` — `RedTeamAssessment` object

**Confidence computation:**
```python
confidence = 1.0 - conformal_width  # Wide interval = low confidence
```

**Decision logic (priority order):**

```
1. Red Team CRITICAL override → always CRITICAL (immediate return)
2. confidence > 0.80 AND risk > 0.60 → FAST-TRACK CRITICAL
   (Novelty 3: skip CBC wait, immediate PCT + culture + vasopressor)
3. confidence < 0.50 → AMBER + uncertainty notification to clinician
4. risk > 0.60 → CRITICAL
5. risk > 0.30 → AMBER
6. else → WATCH
7. [Post-check] Red Team AMBER AND result is WATCH → escalate to AMBER
8. [Post-check] conformal_width > 0.4 AND result is WATCH → escalate to AMBER
```

**Clinical Action Sets:**

| Level | Actions |
|---|---|
| WATCH | Dashboard refresh, reassess in 15 min |
| AMBER | Stat lactate, PCT, blood culture ×2, notify attending, q5min monitoring |
| CRITICAL | Page attending immediately, blood culture, broad-spectrum antibiotics within 1hr, 30mL/kg bolus, CBC/CMP/coag |
| FAST-TRACK | CRITICAL actions PLUS vasopressor preparation, arterial line, skip CBC wait |

**Output:** `OrchestratorDecision` — `{alert_level, risk_score, confidence, conformal_interval, fast_tracked, actions, reasoning, timestamp}`

---

### 6.3 OutcomeLearningAgent — `outcome_learner.py`

**Purpose:** Adaptive feedback loop — learns from resolved cases to improve future alerts.

**Trigger:** 72 hours after an alert, when the actual patient outcome is known.

**Three outcome types tracked:**
- **True Positive:** sepsis occurred AND alert was AMBER/CRITICAL ✅
- **False Negative:** sepsis occurred AND alert was WATCH ❌ (worst case)
- **Near-Miss:** sepsis occurred, model score < 0.3, but Red Team caught it ⚠️

**Near-Miss Response (Novelty 2):**
```python
# Double the loss multiplier for this patient profile
state.loss_multiplier *= 2.0
state.loss_multiplier = min(state.loss_multiplier, 32.0)  # Cap at 32×
```
These multipliers feed back into retraining to penalize similar patient profiles more heavily.

**Adaptive Threshold Update Rule:**
```
threshold_new = threshold_old + η × (target_sensitivity - observed_sensitivity)
η = 0.05 (conservative learning rate)
target_sensitivity = 0.95
```
- If observed sensitivity < 95%: lower thresholds (more aggressive alerting)
- Updates per ICU unit independently (MICU, SICU, CVICU, etc.)
- Minimum 20 resolved cases required before first update

**State stored per ICU unit:** `watch_threshold, amber_threshold, loss_multiplier, total_cases, TP, FN, FP, near_misses`

---

## 7. Baselines (src/baselines/)

### XGBoost Baseline — `xgboost_baseline.py`
- Flattens each `(6, 12)` window to a 72-feature vector
- XGBoost classifier with `scale_pos_weight` for class imbalance
- **Test AUROC: 0.8038** (currently best model)

### SOFA Threshold Baseline — `sofa_baseline.py`
- Clinical scoring system: uses only SOFA components
- Simple threshold: SOFA ≥ 2 → sepsis alert
- **Test AUROC: 0.5869** (barely above random)

---

## 8. Evaluation (src/evaluation/)

### `metrics.py`
Computes:
- **AUROC** — Area Under ROC Curve
- **AUPRC** — Area Under Precision-Recall Curve (more informative for imbalanced data)
- **Sensitivity @ 95% Specificity** — clinically relevant operating point
- **F1 Score**
- **Calibration metrics** — Brier score, reliability diagram data

---

## 9. Three Novel Contributions

### N1 — Quantum-Calibrated Conformal Prediction (QCCP)
- **What:** Uses quantum kernel distance to sepsis centroids as the nonconformity score
- **Why novel:** Classical conformal uses `|y - f(x)|`; quantum kernel captures non-linear structure in Hilbert space, producing tighter intervals
- **Formula:** `s(x) = 1 - max_j K_quantum(x, centroid_j)`
- **Clinical impact:** Tighter intervals → higher confidence → more appropriate fast-tracking

### N2 — Adversarial Tripwire-Gated Asymmetric Safety
- **What:** Red Team Agent (non-overridable) + adaptive loss doubling on near-misses
- **Why novel:** Combines rule-based safety guardrails with learned penalty escalation
- **Feedback loop:** Near-miss → 2× loss multiplier → retraining → better predictions for similar profiles
- **Clinical impact:** System cannot be "convinced" by model confidence to ignore dangerous vitals

### N3 — Confidence-Gated Diagnostic Fast-Tracking
- **What:** When confidence > 80% AND risk > 60%, skip preliminary diagnostics
- **Why novel:** Uses conformal interval width as calibrated proxy for actionable confidence
- **Clinical impact:** Reduces time-to-treatment by bypassing standard diagnostic ordering queue

---

## 10. Full System Data Flow

```
MIMIC-IV CSVs (454M rows)
    ↓ cohort_extraction_optimized.py [chunked, ~1hr]
cohort.csv (94,458 stays, 12,972 sepsis)
    ↓ feature_extraction.py [chunked, ~1.5hrs]
hourly_features.parquet (12 features × hourly bins, ~56 MB)
    ↓ preprocessing.py [~15min]
train/val/test_features.parquet + normalization_stats.json
    ↓ windowing.py [~30min]
features.h5 [(N, 6, 12) tensors — ~4.09M train windows]
    ↓ train_lstm.py [~3hrs on A100]
lstm_best.pt + lstm_embeddings.npz [(N, 16) embeddings]
    ↓ quantum_kernel.py [~30-60min]
quantum_results.json [QSVM trained on 2000 balanced samples]
    ↓ conformal.py [~mins]
ConformalSepsisPredictor [calibrated q_α]
    ↓ [runtime, per patient]
RedTeamAgent → tripwire assessment
ConfidenceGatedOrchestrator → WATCH/AMBER/CRITICAL/FAST-TRACK
    ↓ [72hrs later]
OutcomeLearningAgent → adaptive threshold updates
```

---

## 11. GPU Server Infrastructure

| Item | Value |
|---|---|
| SSH | `ssh csegpuserver@172.16.18.2` |
| GPU 0 | NVIDIA A100-PCIE-40GB ✅ |
| GPU 1 | NVIDIA T400-2GB ❌ (skip) |
| GPU 2 | NVIDIA A100-PCIE-40GB ✅ |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.10.12 (system) — use `pip3 install --user` |
| Project path | `~/QuantumSepsis/` |
| Data path | `~/QuantumSepsis/data/raw/physionet.org/files/mimiciv/3.1/` |
| **Rule** | Always use `screen` — VPN drops kill SSH |
| **GPU select** | `CUDA_VISIBLE_DEVICES=0` or `=2` |

---

## 12. Configuration Reference (`src/config.py`)

All hyperparameters are dataclass-based, loadable from YAML:

```yaml
data:
  bin_size_hours: 1
  forward_fill_limit_hours: 2
  window_size_hours: 6
  window_stride_hours: 1
  prediction_horizon_hours: 4
  val_fraction: 0.15

lstm:
  hidden_dim: 128
  n_layers: 2
  bidirectional: true
  dropout: 0.3
  attention_dim: 64
  embedding_dim: 16

training:
  learning_rate: 0.001
  batch_size: 256
  max_epochs: 100
  early_stopping_patience: 10
  focal_alpha_pos: 0.9
  focal_alpha_neg: 0.1
  focal_gamma: 2.0

quantum:
  n_qubits: 8
  feature_map: ZZFeatureMap
  entanglement: linear
  reps: 2
  pca_components: 8
  backend: aer_simulator
  shots: 1024

red_team:
  temp_low: 36.0
  temp_high: 38.3
  hr_threshold: 90.0
  hr_trend_threshold: 5.0
  rr_threshold: 20.0
  map_threshold: 70.0
  gcs_threshold: 14.0
  critical_tripwire_count: 2

conformal:
  alpha: 0.10
  escalation_width_threshold: 0.40

orchestrator:
  watch_threshold: 0.3
  amber_threshold: 0.6
  high_confidence: 0.80
  low_confidence: 0.50
```

---

## 13. Known Issues & Gotchas

1. **OOM on labevents/chartevents:** Any code calling `load_table()` on these tables will be OOM-killed. Always use `load_table_chunked()` from `cohort_extraction_optimized.py`.

2. **feature_extraction.py still has OOM bug:** Must apply the chunked loading fix to `_load_vitals()` and `_load_labs()` before running on real data (already done on server).

3. **No conda on GPU server:** Only system Python 3.10.12. Install with `pip3 install --user`.

4. **Data path nesting:** wget created nested directories. Actual path is `data/raw/physionet.org/files/mimiciv/3.1/`, NOT `data/raw/mimiciv/3.1/`.

5. **Low AUPRC:** ~0.05 across all models due to severe class imbalance in windowed data (~4M windows with very few positives). AUROC is a more reliable metric here.

6. **Qiskit kernel slow:** Each kernel evaluation requires quantum circuit simulation. 2000×2000 kernel matrix takes ~30-60 min on CPU. Full 4M×4M is impossible.

7. **RBF gamma bug (fixed):** Earlier version recomputed gamma from prediction matrix instead of reusing training gamma → inconsistent kernels → AUROC 0.44. Fixed: always store and reuse `self.rbf_gamma_`.

8. **Shared server:** Check `nvidia-smi` before training. Don't kill other users' screen sessions.

---

## 14. Phase 2 — Pipeline Scripts & Tests (Implemented April 29, 2026)

### Runner Scripts (all support `--synthetic` for local testing)

| Script | What it does | Output |
|--------|-------------|--------|
| `scripts/run_conformal_calibration.py` | Calibrates q_alpha on LSTM val scores, verifies ≥90% coverage on test | `conformal_calibration.json`, `conformal_test_intervals.npz` |
| `scripts/run_e2e_validation.py` | Wires LSTM + Conformal + RedTeam + Orchestrator on test set | `e2e_validation_results.json`, `e2e_decisions.npz` |
| `scripts/run_outcome_learning_simulation.py` | Feeds decisions into OutcomeLearningAgent, tracks FN/near-misses | `outcome_learning_results.json`, `near_miss_weights.json` |
| `scripts/analyze_class_imbalance.py` | Investigates why AUPRC=0.05, focal gamma sensitivity | `class_imbalance_analysis.json` |
| `scripts/run_lstm_tuning.py` | 5 tuning experiments to beat XGBoost 0.8038 | `tuning_results.json` per experiment |

### Test Suite (31 edge case tests)

| Test File | Tests | Covers |
|-----------|-------|--------|
| `tests/test_conformal_calibration.py` | 14 | Coverage guarantee, extreme labels (0%/100%), boundary clipping, batch consistency |
| `tests/test_e2e_validation.py` | 17 | Full pipeline, missing files, wide/zero q_alpha, alert distribution math, norm stats |

### E2E Validation — 5-Step Pipeline

```
Step 1: load_conformal_predictor() + load_norm_stats() + load_model()
Step 2: run_inference() → risk_scores (N,) + windows (N,6,12) + labels (N,)
Step 3: run_red_team() → N RedTeamAssessment objects (CRITICAL/AMBER/WATCH)
Step 4: run_orchestrator() → N OrchestratorDecision objects + alert_labels array
Step 5: compute_metrics() → sensitivity, specificity, F1, AUROC, AUPRC, fn_at_watch
```

### GPU Server Run Order

```bash
# After LSTM training is complete:
python3 scripts/run_conformal_calibration.py        # Step 1
python3 scripts/run_e2e_validation.py               # Step 2
python3 scripts/run_outcome_learning_simulation.py   # Step 3
python3 scripts/analyze_class_imbalance.py           # Step 4
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lstm_tuning.py --exp exp5_combined  # Step 5
```

