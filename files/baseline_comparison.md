# QuantumSepsis Shield — Baseline Comparison Plan

## Experimental Design

All baselines evaluated on **identical MIMIC-IV v3.1 temporal test set** (anchor_year_group = "2020 - 2022").

## Baselines

### 1. SOFA Score Threshold (Clinical Gold Standard)
- **Method:** SOFA ≥ 2 from 24h baseline → sepsis prediction
- **Input:** Raw lab + vitals values
- **Expected AUROC:** 0.65 – 0.70psis
- **Code:** `src/baselines/sofa_baseline.py`
- **Significance:** What clinicians currently use; our floor performance

### 2. NEWS2 Score (Existing Alert System)
- **Method:** National Early Warning Score 2 — aggregate
- **Input:** HR, RR, SpO2, SBP, Temp, Consciousness
- **Expected AUROC:** 0.70 – 0.75
- **Note:** Implemented post-hoc since NEWS2 subsumes some of our features

### 3. XGBoost on Engineered Features
- **Method:** Gradient boosted trees on flattened + statistical features
- **Input:** 72 flattened + 48 statistical + 12 latest = 132 features
- **Expected AUROC:** 0.78 – 0.82
- **Code:** `src/baselines/xgboost_baseline.py`
- **Significance:** Strong classical ML baseline; easy to deploy

### 4. Classical LSTM Only (No Quantum Layer)
- **Method:** Same SepsisLSTM architecture, direct sigmoid output
- **Input:** (6, 12) windows
- **Expected AUROC:** 0.80 – 0.84
- **Code:** `src/training/train_lstm.py` (standalone mode)
- **Significance:** Ablation — shows quantum kernel contribution

### 5. QuantumSepsis Shield (Full System)
- **Method:** LSTM → Quantum Kernel → QSVM → Conformal → Red Team → Orchestrator
- **Input:** (6, 12) windows → 16-dim embedding → 8-dim PCA → quantum kernel
- **Target AUROC:** ≥ 0.85
- **Significance:** Our complete system with all novelties

## Metrics Reported for Each Baseline

| Metric | Description | Target |
|--------|-------------|--------|
| AUROC | Area under ROC curve | ≥ 0.85 |
| AUPRC | Area under precision-recall curve | ≥ 0.50 |
| Sensitivity @ 95% Specificity | Catch rate at low FP rate | Maximize |
| Specificity @ 95% Sensitivity | FP control at target catch rate | Maximize |
| F1 Score | Harmonic mean of precision/recall | Maximize |
| Lead Time | Hours before onset for true positives | 3-4 hours |
| False Negative Rate | Missed sepsis cases | Minimize |
| Calibration (Brier Score) | Probability calibration | Minimize |

## Ablation Studies

| Ablation | Component Removed | Expected Effect |
|----------|-------------------|-----------------|
| A1 | Remove quantum kernel (LSTM only) | AUROC drop 2-5% |
| A2 | Remove Red Team Agent | FN rate increases by 10-30% |
| A3 | Remove conformal prediction | Calibration degrades |
| A4 | Remove confidence-gating | No fast-track, 4-6h delay returns |
| A5 | Remove temporal attention | AUROC drop 1-2% |
| A6 | Remove asymmetric loss | FN rate doubles |

## Actual Results (MIMIC-IV v3.1, Window-Level Prediction)

> **Critical note on AUPRC:** The pipeline does **window-level** prediction at ~1.4% positive rate
> (not stay-level at 13.7%). At 1.4% prevalence, a random classifier AUPRC = 0.014.
> All AUPRC values must be interpreted relative to this random baseline.

| System | AUROC | AUPRC | Sensitivity@95Spec | AUPRC vs Random |
|--------|-------|-------|-------------------|-----------------| 
| SOFA threshold | 0.5869 | 0.0159 | — | 1.1× |
| XGBoost | 0.8038 | 0.0576 | — | 4.1× |
| Classical LSTM | 0.7891 | 0.0519 | 0.2997 | 3.7× |
| **QuantumSepsis Shield** | **TBD** | **TBD** | **TBD** | **TBD** |

> Window shapes: train=4,094,917 | val=729,941 | test=796,893 | positive rate ~1.4%

## Original Expected Results (for reference — written assuming stay-level)

| System | AUROC | AUPRC* | Sensitivity@95Spec | Lead Time (h) |
|--------|-------|--------|-------------------|----------------|
| SOFA threshold | 0.65-0.70 | 0.02-0.04 | 0.15-0.25 | N/A |
| XGBoost | 0.78-0.82 | 0.04-0.07 | 0.40-0.50 | 2-3 |
| Classical LSTM | 0.80-0.84 | 0.04-0.07 | 0.45-0.55 | 3-4 |
| **QuantumSepsis Shield** | **≥ 0.85** | **≥ 0.07** | **≥ 0.55** | **3-4** |

*_AUPRC targets corrected for ~1.4% window-level prevalence._

## Statistical Significance

- **DeLong test** for AUROC comparison between systems
- **McNemar's test** for binary prediction comparison
- **Bootstrap 95% CI** for all metrics (1000 bootstrap samples)
- **p < 0.05** required for claiming improvement over baselines
