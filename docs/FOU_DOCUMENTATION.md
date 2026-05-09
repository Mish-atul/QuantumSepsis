# FOU (Fever of Unknown Origin) Detection System

> **Multi-Class Quantum-Classical System for FOU Categorization**

This document describes the FOU detection extension to QuantumSepsis Shield, demonstrating the scalability of quantum ML approaches to multiple time-critical conditions.

---

## Overview

**FOU (Fever of Unknown Origin)** affects 2-3% of ICU admissions and causes diagnostic delays averaging 2-4 weeks. The FOU detection system categorizes fever cases into:

- **Class 0:** No FOU
- **Class 1:** Infectious FOU (TB, endocarditis, abscess, fungal)
- **Class 2:** Non-infectious FOU (autoimmune, vasculitis, malignancy, drug fever)
- **Class 3:** Undiagnosed FOU

---

## Key Differences from Sepsis Detection

| Aspect | Sepsis | FOU |
|--------|--------|-----|
| **Classification** | Binary (sepsis vs non-sepsis) | Multi-class (4 classes) |
| **Window Size** | 6 hours | 24 hours |
| **Stride** | 1 hour | 6 hours |
| **Prediction Horizon** | 4 hours | 48 hours |
| **Features** | 12 (vitals + labs) | 27 (12 reused + 15 FOU-specific) |
| **Loss Function** | Asymmetric Focal Loss | Multi-class Focal Loss |
| **Conformal Method** | Split Conformal | Adaptive Prediction Sets (APS) |
| **Quantum Kernel** | Binary QSVM | One-vs-Rest QSVM |
| **Red Team Tripwires** | 5 sepsis-specific | 5 FOU-specific |

---

## FOU-Specific Features (15 new features)

| # | Feature | Source | Rationale |
|---|---------|--------|-----------|
| 13 | temp_max_24h | chartevents | Peak fever height |
| 14 | temp_variability | chartevents | Fever pattern (continuous vs intermittent) |
| 15 | fever_duration_hours | chartevents | How long fever has persisted |
| 16 | crp | labevents (50889) | Inflammatory marker |
| 17 | esr | labevents (51288) | Inflammatory marker |
| 18 | procalcitonin | labevents (50963) | Bacterial infection marker |
| 19 | ferritin | labevents (50896) | Inflammatory/malignancy marker |
| 20 | ldh | labevents (50954) | Tissue damage marker |
| 21 | albumin | labevents (50862) | Nutritional/inflammatory status |
| 22 | antibiotic_days | prescriptions | Days on antibiotics |
| 23 | culture_negative_count | microbiologyevents | Number of negative cultures |
| 24 | immunosuppressed | prescriptions | Steroids, chemo, immunosuppressants |
| 25 | weight_loss | chartevents | Weight change (malignancy indicator) |
| 26 | night_sweats_proxy | chartevents | Temperature spikes at night |
| 27 | rash_documented | chartevents | Skin findings (if documented) |

---

## FOU Red Team Tripwires

### 5 Clinical Tripwires

| ID | Name | Condition | Clinical Reason |
|---|---|---|---|
| **TW-FEVER** | Persistent Fever | Temp > 39°C for > 72 hours | Severe persistent fever |
| **TW-SEPSIS** | Sepsis Risk | qSOFA ≥ 2 | Rule out sepsis first |
| **TW-NEUTROPENIA** | Neutropenia | ANC < 500 | High infection risk |
| **TW-HYPOTENSION** | Hypotension | MAP < 65 mmHg | Hemodynamic instability |
| **TW-ALTERED** | Altered Mental Status | GCS < 13 | CNS involvement |

### Escalation Logic

- **≥ 2 tripwires** → CRITICAL (immediate infectious disease consult)
- **1 tripwire** → AMBER (expedite workup)
- **0 tripwires** → WATCH (routine monitoring)

---

## Unified Orchestrator Priority Logic

The unified orchestrator handles both sepsis and FOU with the following priority:

1. **Sepsis takes priority** (more acute, life-threatening)
2. **If sepsis risk < 0.3 AND FOU probability > 0.5** → FOU workup
3. **If both high** → treat sepsis first, then FOU workup
4. **Red Team overrides** apply to both conditions

### Decision Flow

```
Input: sepsis_risk, fou_probabilities[4]

IF red_team_critical:
    → CRITICAL (override)
ELIF sepsis_risk >= 0.3:
    → Prioritize sepsis
    IF fou_prob > 0.5:
        → Add "After sepsis stabilization, initiate FOU workup"
ELIF fou_prob > 0.5:
    → FOU workup (sepsis risk low)
ELIF sepsis_risk >= 0.3 OR fou_prob >= 0.5:
    → WATCH both conditions
ELSE:
    → WATCH (neither condition detected)
```

---

## Multi-Class Conformal Prediction

### Adaptive Prediction Sets (APS)

FOU uses **Adaptive Prediction Sets** (Romano et al., 2020) for multi-class conformal prediction:

1. Sort classes by predicted probability (descending)
2. Include classes until cumulative probability ≥ 1-α
3. Guarantees: P(y_true ∈ prediction_set) ≥ 1-α

**Example:**
- Predicted probabilities: [0.1, 0.6, 0.25, 0.05]
- α = 0.10 (90% coverage)
- Sorted: [1: 0.6, 2: 0.25, 0: 0.1, 3: 0.05]
- Cumulative: [0.6, 0.85, 0.95, 1.0]
- Prediction set: [1, 2] (Infectious or Non-infectious FOU)

### QCCP for Multi-Class

Quantum-Calibrated Conformal Prediction extends to multi-class:

- **Centroids:** One per class (4 total)
- **Nonconformity score:** s(x) = 1 - max_j K_quantum(x, centroid_j)
- **Advantage:** Tighter prediction sets using quantum kernel distances

---

## Training Pipeline

### Step 1: Cohort Extraction
```bash
python3 -m src.data.cohort_extraction_fou \
    data/raw/mimiciv/3.1 \
    data/processed/fou
```

**Inclusion Criteria:**
- ICU stay ≥ 7 days
- Fever > 38.3°C on ≥ 3 occasions
- Negative initial cultures
- No obvious infection source

**Expected Cohort:** 2,000-3,000 FOU cases from 94,458 ICU stays

### Step 2: Feature Engineering
```bash
python3 -m src.data.feature_engineering_fou \
    data/raw/mimiciv/3.1 \
    data/processed/fou/fou_cohort.csv \
    data/processed/fou
```

**Output:** 27 features per hour

### Step 3: Preprocessing
```bash
python3 -m src.data.preprocessing_fou \
    data/processed/fou/fou_hourly_features.parquet \
    data/processed/fou/fou_cohort.csv \
    data/processed/fou
```

**Steps:**
1. Forward-fill (6-hour limit)
2. Median imputation
3. Z-score normalization
4. Temporal split (train/val/test)

### Step 4: Windowing
```bash
python3 -m src.data.windowing_fou \
    data/processed/fou/fou_train_features.parquet \
    data/processed/fou/fou_val_features.parquet \
    data/processed/fou/fou_test_features.parquet \
    data/processed/fou/fou_features.h5
```

**Output:** HDF5 with shape (N, 24, 27)

### Step 5: LSTM Training
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_fou_lstm.py \
    --data data/processed/fou/fou_features.h5 \
    --epochs 100 \
    --batch-size 256
```

**Model:** BiLSTM + Attention (~450K parameters)
**Loss:** Multi-class Focal Loss (α=[0.1, 0.4, 0.3, 0.2])
**Metric:** Macro F1 (early stopping)

### Step 6: Conformal Calibration
```bash
python3 scripts/calibrate_fou_conformal.py \
    --model checkpoints/fou/fou_lstm_best.pt \
    --data data/processed/fou/fou_features.h5
```

**Method:** Adaptive Prediction Sets
**Coverage:** ≥90%

### Step 7: Quantum Kernel Training
```bash
python3 scripts/train_fou_quantum.py \
    --embeddings data/processed/fou/fou_lstm_embeddings.npz \
    --max-samples 2000
```

**Architecture:**
- 8 qubits (PCA: 16-dim → 8-dim)
- ZZFeatureMap, linear entanglement, 2 reps
- One-vs-Rest QSVM (4 binary classifiers)
- Balanced subsample: 500 per class × 4 = 2000 samples

### Step 8: E2E Validation
```bash
python3 scripts/run_fou_e2e_validation.py \
    --model checkpoints/fou/fou_lstm_best.pt \
    --data data/processed/fou/fou_features.h5
```

**Pipeline:**
1. LSTM inference → probabilities
2. Conformal prediction → prediction sets
3. Red Team assessment → tripwire checks
4. Unified orchestrator → final decision

---

## Expected Results

### Performance Targets

| Model | Macro F1 | Infectious AUROC | Non-infectious AUROC |
|-------|----------|------------------|----------------------|
| LSTM | 0.70 | 0.80 | 0.75 |
| Quantum Kernel | 0.65 | 0.75 | 0.70 |
| Ensemble | 0.72 | 0.82 | 0.77 |

### Conformal Prediction

- **Coverage:** ≥90% (guaranteed)
- **Average set size:** 1.5-2.0 classes
- **Singleton rate:** 60-70%
- **QCCP advantage:** 30-40% smaller prediction sets vs classical

### Clinical Impact

- **Time to diagnosis:** 30-40% reduction (from 2-4 weeks to 1-2 weeks)
- **Unnecessary antibiotics:** 20-30% reduction
- **Diagnostic yield:** 70-80% correct categorization
- **False alarm rate:** <15%

---

## Implementation Files

### Data Pipeline
- `src/data/cohort_extraction_fou.py` — FOU cohort extraction
- `src/data/feature_engineering_fou.py` — 27 feature extraction
- `src/data/preprocessing_fou.py` — Preprocessing with 6-hour forward-fill
- `src/data/windowing_fou.py` — 24-hour windows, 6-hour stride

### Models
- `src/models/lstm_fou.py` — Multi-class BiLSTM + Attention
- `src/models/losses.py` — Multi-class Focal Loss (extended)
- `src/models/quantum_kernel_fou.py` — One-vs-Rest QSVM
- `src/models/conformal_fou.py` — APS + QCCP for multi-class

### Agents
- `src/agents/red_team_fou.py` — FOU-specific tripwires
- `src/agents/orchestrator_unified.py` — Multi-condition orchestrator

### Training Scripts
- `scripts/train_fou_lstm.py` — LSTM training
- `scripts/calibrate_fou_conformal.py` — Conformal calibration
- `scripts/train_fou_quantum.py` — Quantum kernel training
- `scripts/run_fou_e2e_validation.py` — E2E validation

### Configuration
- `src/config.py` — Extended with FouConfig, UnifiedConfig

---

## Integration with Sepsis System

The FOU system is designed to work **independently** and **together** with the sepsis detection system:

### Independent Operation
- Separate data pipelines (different processed directories)
- Separate models (different checkpoints)
- Separate configurations (FouConfig vs Config)
- Can be trained and deployed independently

### Unified Operation
- **Unified Orchestrator** coordinates both systems
- **Priority logic** ensures sepsis takes precedence
- **Shared infrastructure** (MIMIC-IV, GPU server, quantum backend)
- **Consistent architecture** (BiLSTM + Quantum + Conformal + Red Team)

---

## GPU Server Compatibility

The FOU system is fully compatible with the existing GPU server setup:

- **CUDA support:** Uses same PyTorch/CUDA stack as sepsis
- **Data location:** Separate `data/processed/fou/` directory
- **Checkpoints:** Separate `checkpoints/fou/` directory
- **No conflicts:** Can coexist with sepsis system

---

## Next Steps

1. **Run cohort extraction** on MIMIC-IV to verify cohort size (target: ≥1,500 cases)
2. **Train FOU LSTM** on GPU server (expected: 2-3 hours on A100)
3. **Evaluate performance** against targets (Macro F1 ≥ 0.70)
4. **Train quantum kernel** (expected: 40-50 minutes)
5. **Validate E2E pipeline** with unified orchestrator
6. **Compare with baselines** (XGBoost, classical conformal)

---

## References

1. Petersdorf RG, Beeson PB. Fever of unexplained origin. *Medicine* 1961;40:1-30.
2. Romano Y, et al. Classification with Valid and Adaptive Coverage. *NeurIPS* 2020.
3. Lin TY, et al. Focal Loss for Dense Object Detection. *ICCV* 2017.
4. Schuld M, Killoran N. Quantum Machine Learning in Feature Hilbert Spaces. *PRL* 2019;122:040504.

---

**Status:** ✅ Implementation Complete — Ready for Training on GPU Server
