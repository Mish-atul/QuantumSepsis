# QuantumSepsis Shield

> **Adversarially-Safe Quantum-Classical System for Multi-Condition Early Warning**

QuantumSepsis Shield is a unified platform for early detection of time-critical conditions in ICU patients. Initially focused on sepsis detection (11 million deaths/year globally), the system now extends to **Fever of Unknown Origin (FOU)** detection, demonstrating scalability of quantum ML approaches to multiple conditions.

**Sepsis Detection:** 3-4 hours before clinical onset  
**FOU Detection:** Multi-class categorization (Infectious, Non-infectious, Undiagnosed)

## Architecture

### Sepsis Detection Pipeline
```
Layer 1: Monitoring Agents     → Real-time data ingestion, 6-hour windowing
Layer 2: BiLSTM Encoder        → Temporal encoding → 16-dim latent embedding
Layer 3: Quantum Kernel        → ZZFeatureMap (8 qubits) → QSVM classification
Layer 4a: Conformal Prediction → 90% coverage uncertainty intervals
Layer 4b: Red Team Agent       → Non-overridable clinical tripwires (5 rules)
Layer 5: Orchestrator          → Confidence-gated decision → WATCH/AMBER/CRITICAL/FAST-TRACK
```

### FOU Detection Pipeline
```
Layer 1: Monitoring Agents     → 24-hour windowing, 27 features
Layer 2: BiLSTM Encoder        → Multi-class temporal encoding → 16-dim embedding
Layer 3: Quantum Kernel        → One-vs-Rest QSVM (4 classes)
Layer 4a: Conformal Prediction → Adaptive Prediction Sets (APS)
Layer 4b: Red Team Agent       → FOU-specific tripwires (persistent fever, qSOFA, neutropenia)
Layer 5: Unified Orchestrator  → Multi-condition priority logic
```

## Current Results

| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec |
|-------|-----------|------------|---------------------|
| SOFA Threshold | 0.5869 | 0.0159 | — |
| LSTM V1 Improved (39 features) | 0.7905 | 0.0531 | 0.3045 |
| XGBoost Baseline (132 features) | 0.8038 | 0.0576 | 0.3169 |
| **Ensemble (30% LSTM + 70% XGBoost)** | **0.8051** | **0.0581** | **0.3213** |
| **Qiskit Quantum Kernel (8 qubits)** | **0.7598** | **0.0365** | **0.2138** |

### Quantum Advantage (Sepsis)
- **QCCP Width Reduction:** 55.8% tighter uncertainty intervals vs classical conformal
- **Training Time:** 41.4 minutes (500×500 quantum kernel matrix)
- **Support Vectors:** 395/500 (79% efficiency)

## FOU Detection Results

| Model | Macro F1 | Infectious AUROC | Non-infectious AUROC |
|-------|----------|------------------|----------------------|
| **Target (LSTM)** | **0.70** | **0.80** | **0.75** |
| **Target (Quantum)** | **0.65** | **0.75** | **0.70** |

### FOU Implementation Status
- ✅ Configuration extended for FOU (27 features, 24-hour windows, 4-class output)
- ✅ Data pipeline (cohort extraction, feature engineering, preprocessing, windowing)
- ✅ Multi-class LSTM with Focal Loss
- ✅ Multi-class conformal prediction (Adaptive Prediction Sets)
- ✅ FOU-specific Red Team Agent (5 clinical tripwires)
- ✅ Quantum kernel with One-vs-Rest QSVM
- ✅ Unified orchestrator for multi-condition detection
- ✅ Training scripts for full FOU pipeline
- 🔄 **Ready for training on GPU server**

## Current Status

| Stage | Status |
|-------|--------|
| Code implementation (24 source modules) | ✅ Complete |
| MIMIC-IV v3.1 download (9.9 GB) | ✅ Complete |
| Cohort extraction (94,458 ICU stays, 12,972 sepsis) | ✅ Complete |
| Feature extraction (12 features/hour) | ✅ Complete |
| Preprocessing + Windowing (4.09M train windows) | ✅ Complete |
| LSTM V1 Improved training (39 features, AUROC 0.7905) | ✅ Complete |
| XGBoost Baseline (132 features, AUROC 0.8038) | ✅ Complete |
| Ensemble Model (AUROC 0.8051) | ✅ Complete |
| Embedding extraction (16-dim, all splits) | ✅ Complete |
| **Quantum Kernel (Qiskit, 8 qubits, AUROC 0.7598)** | ✅ Complete |
| **QCCP (55.8% width reduction)** | ✅ Complete |
| **Conformal calibration (99.46% coverage)** | ✅ Complete |
| **E2E validation (796,893 test windows)** | ✅ Complete |
| **Outcome learning simulation** | ✅ Complete |
| **Phase 2 Pipeline (Conformal + Safety + Orchestrator)** | ✅ Complete |
| **Unisys Innovation Program Materials** | ✅ Complete |
| **FOU Extension Implementation** | ✅ Complete |

## Quick Start

### GPU Server — Sepsis Pipeline (Complete)
```bash
ssh csegpuserver@172.16.18.2
cd ~/QuantumSepsis

# Phase 1: LSTM V1 Improved Training (✅ Complete)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_v1_improved.py

# Phase 1: XGBoost Baseline (✅ Complete)
python3 scripts/train_xgboost_real.py

# Phase 1: Ensemble Evaluation (✅ Complete)
python3 scripts/evaluate_real_ensemble.py

# Phase 2: Conformal Calibration (✅ Complete)
python3 scripts/run_ensemble_conformal.py

# Phase 2: E2E Validation (✅ Complete)
python3 scripts/run_ensemble_e2e_validation.py

# Phase 3: Quantum Kernel (✅ Complete)
python3 scripts/run_quantum_fixed.py

# Phase 3: QCCP (✅ Complete)
python3 scripts/run_qccp.py

# Generate Unisys Report
python3 scripts/generate_quantum_report.py
```

### GPU Server — FOU Pipeline (Ready for Training)
```bash
ssh csegpuserver@172.16.18.2
cd ~/QuantumSepsis

# Step 1: Extract FOU cohort from MIMIC-IV
python3 -m src.data.cohort_extraction_fou \
    data/raw/mimiciv/3.1 \
    data/processed/fou

# Step 2: Extract FOU features (27 features)
python3 -m src.data.feature_engineering_fou \
    data/raw/mimiciv/3.1 \
    data/processed/fou/fou_cohort.csv \
    data/processed/fou

# Step 3: Preprocess features
python3 -m src.data.preprocessing_fou \
    data/processed/fou/fou_hourly_features.parquet \
    data/processed/fou/fou_cohort.csv \
    data/processed/fou

# Step 4: Create windows (24-hour, 6-hour stride)
python3 -m src.data.windowing_fou \
    data/processed/fou/fou_train_features.parquet \
    data/processed/fou/fou_val_features.parquet \
    data/processed/fou/fou_test_features.parquet \
    data/processed/fou/fou_features.h5

# Step 5: Train FOU LSTM (multi-class)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_fou_lstm.py \
    --data data/processed/fou/fou_features.h5 \
    --epochs 100 \
    --batch-size 256

# Step 6: Calibrate conformal prediction
python3 scripts/calibrate_fou_conformal.py \
    --model checkpoints/fou/fou_lstm_best.pt \
    --data data/processed/fou/fou_features.h5

# Step 7: Train quantum kernel (One-vs-Rest)
python3 scripts/train_fou_quantum.py \
    --embeddings data/processed/fou/fou_lstm_embeddings.npz \
    --max-samples 2000

# Step 8: E2E validation
python3 scripts/run_fou_e2e_validation.py \
    --model checkpoints/fou/fou_lstm_best.pt \
    --data data/processed/fou/fou_features.h5
```

### Local — Smoke Tests (no GPU or data needed)
```bash
python3 scripts/run_conformal_calibration.py --synthetic
python3 scripts/run_e2e_validation.py --synthetic
python3 scripts/run_outcome_learning_simulation.py --synthetic
```

### Run Tests
```bash
python3 tests/test_conformal_calibration.py
python3 tests/test_e2e_validation.py
```

## Project Structure

```
QuantumSepsis/
├── src/
│   ├── config.py                           # Unified configuration (Sepsis + FOU + Unified)
│   ├── data/                               # Data pipeline (Sepsis + FOU)
│   │   ├── cohort_extraction_optimized.py  # Sepsis cohort extraction
│   │   ├── cohort_extraction_fou.py        # FOU cohort extraction (NEW)
│   │   ├── feature_extraction.py           # Sepsis feature extraction (12 features)
│   │   ├── feature_engineering_fou.py      # FOU feature extraction (27 features) (NEW)
│   │   ├── preprocessing.py                # Sepsis preprocessing
│   │   ├── preprocessing_fou.py            # FOU preprocessing (NEW)
│   │   ├── windowing.py                    # Sepsis windowing (6-hour)
│   │   ├── windowing_fou.py                # FOU windowing (24-hour) (NEW)
│   │   └── dataset.py                      # PyTorch DataLoaders
│   ├── models/                             # Models (Sepsis + FOU)
│   │   ├── lstm.py                         # Sepsis BiLSTM (binary)
│   │   ├── lstm_fou.py                     # FOU BiLSTM (multi-class) (NEW)
│   │   ├── losses.py                       # Focal Loss + Multi-class Focal Loss (UPDATED)
│   │   ├── ensemble_lstm_xgb.py            # Sepsis ensemble
│   │   ├── quantum_kernel.py               # Sepsis quantum kernel
│   │   ├── quantum_kernel_fou.py           # FOU quantum kernel (One-vs-Rest) (NEW)
│   │   ├── conformal.py                    # Sepsis conformal prediction
│   │   └── conformal_fou.py                # FOU conformal (APS + QCCP) (NEW)
│   ├── agents/                             # Safety agents (Sepsis + FOU + Unified)
│   │   ├── red_team.py                     # Sepsis red team (5 tripwires)
│   │   ├── red_team_fou.py                 # FOU red team (5 FOU-specific tripwires) (NEW)
│   │   ├── orchestrator.py                 # Sepsis orchestrator
│   │   ├── orchestrator_unified.py         # Unified multi-condition orchestrator (NEW)
│   │   └── outcome_learner.py              # Adaptive threshold tuning
│   ├── baselines/                          # Baselines
│   │   ├── sofa_baseline.py                # SOFA score baseline
│   │   └── xgboost_baseline.py             # XGBoost baseline
│   └── evaluation/metrics.py               # Evaluation metrics
├── scripts/                                # Training scripts (Sepsis + FOU)
│   ├── train_v1_improved.py                # Sepsis LSTM training
│   ├── train_fou_lstm.py                   # FOU LSTM training (NEW)
│   ├── train_xgboost_real.py               # Sepsis XGBoost training
│   ├── evaluate_real_ensemble.py           # Sepsis ensemble evaluation
│   ├── run_ensemble_conformal.py           # Sepsis conformal calibration
│   ├── calibrate_fou_conformal.py          # FOU conformal calibration (NEW)
│   ├── run_ensemble_e2e_validation.py      # Sepsis E2E validation
│   ├── run_fou_e2e_validation.py           # FOU E2E validation (NEW)
│   ├── run_quantum_fixed.py                # Sepsis quantum kernel
│   ├── train_fou_quantum.py                # FOU quantum kernel (NEW)
│   ├── run_qccp.py                         # Sepsis QCCP
│   └── generate_quantum_report.py          # Unisys report generator
├── docs/                                   # Documentation
│   ├── FOU_IMPLEMENTATION_PLAN.md          # FOU implementation plan
│   ├── FOU_DOCUMENTATION.md                # FOU system documentation (NEW)
│   ├── UNISYS_PRESENTATION_SUMMARY.md      # Unisys presentation guide
│   ├── FINAL_RESULTS.md                    # Final results summary
│   └── IMPLEMENTATION_PLAN.md              # Sepsis implementation plan
├── data/                                   # Data directories
│   ├── raw/mimiciv/3.1/                    # MIMIC-IV v3.1 raw data
│   ├── processed/sepsis/                   # Sepsis processed data
│   └── processed/fou/                      # FOU processed data (NEW)
├── checkpoints/                            # Model checkpoints
│   ├── sepsis/                             # Sepsis model checkpoints
│   └── fou/                                # FOU model checkpoints (NEW)
└── requirements.txt                        # Python dependencies
```

## Dataset

**MIMIC-IV v3.1** — 364,627 patients, 94,458 ICU stays, ~454M total event rows.

| Feature | Source | Items |
|---------|--------|-------|
| Heart Rate, SBP, DBP, MAP, Temp, RR, SpO2, GCS | chartevents (~330M rows) | 35 item IDs |
| Lactate, WBC, Creatinine, Platelets | labevents (~124M rows) | 4 item IDs |

## Three Novelties

1. **Quantum-Calibrated Conformal Prediction (QCCP)** — uncertainty from quantum Hilbert space distance
2. **Adversarial Tripwire-Gated Safety** — Red Team Agent (5 clinical tripwires) + adaptive loss doubling from near-misses
3. **Confidence-Gated Diagnostic Fast-Tracking** — skip preliminary labs when conformal confidence > 80%

## E2E Pipeline Flow

```
MIMIC-IV CSVs (454M rows)
    ↓ cohort_extraction_optimized.py
cohort.csv (94,458 stays, 12,972 sepsis)
    ↓ feature_extraction.py
hourly_features.parquet (12 features × hourly bins)
    ↓ preprocessing.py
train/val/test_features.parquet + normalization_stats.json
    ↓ windowing.py
features.h5 (4.09M train windows, shape N×6×12)
    ↓ train_v1_improved.py
lstm_v1_improved_best.pt + lstm_embeddings.npz (16-dim)
    ↓ train_xgboost_real.py
xgboost_baseline.pkl (132 features)
    ↓ evaluate_real_ensemble.py
Ensemble AUROC: 0.8051 (30% LSTM + 70% XGBoost)
    ↓ run_ensemble_conformal.py
conformal_calibration.json (q_alpha: 0.2663, coverage: 99.46%)
    ↓ run_ensemble_e2e_validation.py
    ├── Ensemble inference → risk scores
    ├── ConformalSepsisPredictor → [lower, upper] intervals
    ├── RedTeamAgent → tripwire assessments (49.91% overrides)
    └── Orchestrator → WATCH/AMBER/CRITICAL decisions
    ↓ run_quantum_fixed.py
quantum_results_qiskit.json (AUROC: 0.7598, 8 qubits)
    ↓ run_qccp.py
qccp_results.json (55.8% width reduction vs classical)
    ↓ generate_quantum_report.py
quantum_advantage_report_unisys.json (Unisys Innovation Program)
```

## Team

- **Yash Gautam** — Data Pipeline + GPU Training + Phase 2 Integration
- **Atul Kumar Mishra (AKM)** — Model Architecture + Quantum Kernel
- **Tanishk Viraj Bhanage** — Safety Agents + Evaluation

## References

- Singer et al. (2016). Sepsis-3 Definitions. *JAMA* 315(8):801-810.
- Havlíček et al. (2019). Quantum-enhanced feature spaces. *Nature* 567:209-212.
- Johnson et al. (2023). MIMIC-IV. *PhysioNet*. DOI: 10.13026/6mm1-ek67.

## License

Research use only. MIMIC-IV PhysioNet Data Use Agreement applies.
