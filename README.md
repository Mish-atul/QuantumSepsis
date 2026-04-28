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
| **LSTM (BiLSTM+Attention)** | **0.7891** | **0.0519** | **0.2997** |
| **XGBoost** | **0.8038** | **0.0576** | — |
| RBF Quantum Kernel (proxy) | 0.7879 | 0.0520 | 0.2999 |
| Qiskit Quantum Kernel | 🔄 Pending | — | — |

## Current Status

| Stage | Status |
|-------|--------|
| Code implementation (24 source modules) | ✅ Complete |
| MIMIC-IV v3.1 download (9.9 GB) | ✅ Complete |
| Cohort extraction (94,458 ICU stays, 12,972 sepsis) | ✅ Complete |
| Feature extraction (12 features/hour) | ✅ Complete |
| Preprocessing + Windowing (4.09M train windows) | ✅ Complete |
| LSTM training (A100 GPU, val AUROC 0.7601) | ✅ Complete |
| Embedding extraction (16-dim, all splits) | ✅ Complete |
| Baselines (XGBoost + SOFA) | ✅ Complete |
| **Conformal calibration script** | ✅ Complete |
| **E2E orchestrator validation script** | ✅ Complete |
| **Outcome learning simulation script** | ✅ Complete |
| **Class imbalance analysis script** | ✅ Complete |
| **LSTM hyperparameter tuning script** | ✅ Complete |
| **Edge case tests (31 tests)** | ✅ Complete |
| Quantum kernel integration (Qiskit) | 🔄 In Progress |
| QCCP (quantum-calibrated conformal) | ⏳ Blocked by quantum kernel |

## Quick Start

### GPU Server — Full Pipeline
```bash
ssh csegpuserver@172.16.18.2
cd ~/QuantumSepsis

# Phase 1: Data pipeline + LSTM training (already done)
python3 -m src.data.cohort_extraction_optimized --data-dir data/raw/physionet.org/files/mimiciv/3.1
python3 -m src.data.feature_extraction --cohort data/processed/cohort.csv --data-dir data/raw/physionet.org/files/mimiciv/3.1
python3 -m src.data.preprocessing --features data/processed/hourly_features.parquet --cohort data/processed/cohort.csv
python3 scripts/run_windowing_real.py
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm --data data/processed/features.h5
python3 scripts/run_real_baselines.py

# Phase 2: Conformal calibration + E2E validation (NEW)
python3 scripts/run_conformal_calibration.py
python3 scripts/run_e2e_validation.py
python3 scripts/run_outcome_learning_simulation.py
python3 scripts/analyze_class_imbalance.py

# Phase 2: LSTM tuning (run best experiment)
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lstm_tuning.py --exp exp5_combined
```

### Local — Smoke Tests (no GPU or data needed)
```bash
python3 scripts/run_conformal_calibration.py --synthetic
python3 scripts/run_e2e_validation.py --synthetic
python3 scripts/run_outcome_learning_simulation.py --synthetic
python3 scripts/analyze_class_imbalance.py --synthetic
python3 scripts/run_lstm_tuning.py --synthetic
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
│   ├── data/                               # Data pipeline (6 modules)
│   │   ├── cohort_extraction_optimized.py  # Memory-safe Sepsis-3 extraction
│   │   ├── feature_extraction.py           # 12 vitals/labs per hour
│   │   ├── preprocessing.py                # Imputation + normalization
│   │   ├── windowing.py                    # 6-hour sliding windows → HDF5
│   │   └── dataset.py                      # PyTorch DataLoaders
│   ├── models/                             # 4 model modules
│   │   ├── lstm.py                         # BiLSTM + Temporal Attention (~420K params)
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
│   ├── run_conformal_calibration.py        # LSTM → q_alpha → conformal intervals
│   ├── run_e2e_validation.py               # LSTM + Conformal + RedTeam + Orchestrator
│   ├── run_outcome_learning_simulation.py  # Adaptive feedback loop on test decisions
│   ├── analyze_class_imbalance.py          # AUPRC investigation + recommendations
│   ├── run_lstm_tuning.py                  # 5 hyperparameter experiments
│   ├── run_windowing_real.py               # Real-data windowing runner
│   ├── run_real_baselines.py               # Baseline metric comparison
│   └── run_pipeline_autonomous.sh          # Full pipeline shell orchestrator
├── tests/                                  # Edge case tests
│   ├── test_conformal_calibration.py       # 14 tests (coverage, boundaries, edge cases)
│   └── test_e2e_validation.py              # 17 tests (full pipeline, missing files, etc.)
├── files/                                  # Documentation
│   ├── architecture.md                     # Full 5-layer technical spec
│   ├── dataset.md                          # MIMIC-IV v3.1 reference
│   ├── novelty.md                          # 3 novel contributions
│   └── roadmap.md                          # Execution plan
├── agents.md                               # Complete project knowledge base
├── backlog.md                              # Done vs pending task tracker
├── PROGRESS_REPORT.md                      # Progress + teammate handoff
├── IMPLEMENTATION_PLAN.md                  # Detailed execution plan
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
    ↓ train_lstm.py
lstm_best.pt + lstm_embeddings.npz (16-dim)
    ↓ run_conformal_calibration.py
conformal_calibration.json (q_alpha threshold)
    ↓ run_e2e_validation.py
    ├── LSTM inference → risk scores
    ├── ConformalSepsisPredictor → [lower, upper] intervals
    ├── RedTeamAgent → tripwire assessments
    └── Orchestrator → WATCH/AMBER/CRITICAL/FAST-TRACK decisions
    ↓ run_outcome_learning_simulation.py
outcome_learning_results.json (adaptive thresholds + near-miss weights)
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
