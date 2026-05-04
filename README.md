# QuantumSepsis Shield

> **Adversarially-Safe Quantum-Classical System for Early Sepsis Detection**

Sepsis kills 11 million people per year globally. QuantumSepsis Shield detects sepsis **3-4 hours before clinical onset** using a 5-layer agentic pipeline combining classical deep learning, quantum kernel methods, conformal prediction, and adversarial safety mechanisms.

## Architecture

```
Layer 1: Monitoring Agents     → Real-time data ingestion, 6-hour windowing
Layer 2: BiLSTM Encoder        → Temporal encoding → 16-dim latent embedding
Layer 3: Quantum Kernel        → ZZFeatureMap (8 qubits) → QSVM classification
Layer 4a: Conformal Prediction → 90% coverage uncertainty intervals
Layer 4b: Red Team Agent       → Non-overridable clinical tripwires (5 rules)
Layer 5: Orchestrator          → Confidence-gated decision → WATCH/AMBER/CRITICAL/FAST-TRACK
```

## Current Results

| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec |
|-------|-----------|------------|---------------------|
| SOFA Threshold | 0.5869 | 0.0159 | — |
| LSTM V1 Improved (39 features) | 0.7905 | 0.0531 | 0.3045 |
| XGBoost Baseline (132 features) | 0.8038 | 0.0576 | 0.3169 |
| **Ensemble (30% LSTM + 70% XGBoost)** | **0.8051** | **0.0581** | **0.3213** |
| **Qiskit Quantum Kernel (8 qubits)** | **0.7598** | **0.0365** | **0.2138** |

### Quantum Advantage
- **QCCP Width Reduction:** 55.8% tighter uncertainty intervals vs classical conformal
- **Training Time:** 41.4 minutes (500×500 quantum kernel matrix)
- **Support Vectors:** 395/500 (79% efficiency)

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

## Quick Start

### GPU Server — Full Pipeline
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
│   ├── config.py                           # All hyperparameters (dataclass-based)
│   ├── data/                               # Data pipeline (7 modules)
│   │   ├── cohort_extraction_optimized.py  # Memory-safe Sepsis-3 extraction
│   │   ├── feature_extraction.py           # 12 vitals/labs per hour
│   │   ├── feature_engineering_v1_enhanced.py # 39 features (12 raw + 27 derived)
│   │   ├── preprocessing.py                # Imputation + normalization
│   │   ├── windowing.py                    # 6-hour sliding windows → HDF5
│   │   └── dataset.py                      # PyTorch DataLoaders
│   ├── models/                             # 5 model modules
│   │   ├── lstm.py                         # BiLSTM + Temporal Attention (~420K params)
│   │   ├── ensemble_lstm_xgb.py            # Ensemble (30% LSTM + 70% XGBoost)
│   │   ├── losses.py                       # Asymmetric Focal Loss (FN=9×FP)
│   │   ├── quantum_kernel.py               # ZZFeatureMap + QSVM (Qiskit)
│   │   └── conformal.py                    # Split Conformal + QCCP
│   ├── training/train_lstm.py              # Full training with early stopping + W&B
│   ├── agents/                             # Safety agents (3 modules)
│   │   ├── red_team.py                     # 5 non-overridable clinical tripwires
│   │   ├── orchestrator.py                 # Confidence-gated alert fusion
│   │   └── outcome_learner.py              # Adaptive threshold tuning + near-miss feedback
│   ├── baselines/                          # XGBoost + SOFA comparisons
│   └── evaluation/metrics.py               # AUROC, AUPRC, calibration
├── scripts/                                # Pipeline runner scripts
│   ├── train_v1_improved.py                # LSTM V1 Improved training (39 features)
│   ├── train_xgboost_real.py               # XGBoost baseline training
│   ├── evaluate_real_ensemble.py           # Ensemble evaluation
│   ├── run_ensemble_conformal.py           # Ensemble conformal calibration
│   ├── run_ensemble_e2e_validation.py      # Ensemble E2E validation
│   ├── run_quantum_fixed.py                # Quantum kernel training (Qiskit)
│   ├── run_qccp.py                         # Quantum-calibrated conformal prediction
│   ├── generate_quantum_report.py          # Unisys Innovation Program report
│   ├── run_conformal_calibration.py        # LSTM → q_alpha → conformal intervals
│   ├── run_e2e_validation.py               # LSTM + Conformal + RedTeam + Orchestrator
│   ├── run_outcome_learning_simulation.py  # Adaptive feedback loop on test decisions
│   ├── run_windowing_real.py               # Real-data windowing runner
│   └── run_real_baselines.py               # Baseline metric comparison
├── tests/                                  # Edge case tests
│   ├── test_conformal_calibration.py       # 14 tests (coverage, boundaries, edge cases)
│   └── test_e2e_validation.py              # 17 tests (full pipeline, missing files, etc.)
├── docs/                                   # Documentation
│   ├── UNISYS_PRESENTATION_SUMMARY.md      # Complete Unisys presentation guide
│   ├── UNISYS_EXECUTIVE_SUMMARY.txt        # One-page executive summary
│   ├── RESULT_FILES_GUIDE.md               # Guide to all result files
│   ├── FINAL_RESULTS.md                    # Final results summary
│   ├── IMPLEMENTATION_PLAN.md              # Detailed execution plan
│   └── MODEL_IMPROVEMENT.md                # Model improvement strategies
├── files/                                  # Technical documentation
│   ├── architecture.md                     # Full 5-layer technical spec
│   ├── dataset.md                          # MIMIC-IV v3.1 reference
│   ├── novelty.md                          # 3 novel contributions
│   └── roadmap.md                          # Execution plan
├── agents.md                               # Complete project knowledge base
├── PROGRESS_REPORT.md                      # Progress + teammate handoff
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
