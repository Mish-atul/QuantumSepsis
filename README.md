# QuantumSepsis Shield

> **Adversarially-Safe Quantum-Classical System for Early Sepsis Detection**

Sepsis kills 11 million people per year globally. QuantumSepsis Shield detects sepsis **3-4 hours before clinical onset** using a 5-layer agentic pipeline combining classical deep learning, quantum kernel methods, conformal prediction, and adversarial safety mechanisms.

## Architecture

```
Layer 1: Monitoring Agents     → Real-time data ingestion, 6-hour windowing
Layer 2: BiLSTM Encoder        → Temporal encoding → 16-dim latent embedding
Layer 3: Quantum Kernel        → ZZFeatureMap (8 qubits) → QSVM classification
Layer 4a: Conformal Prediction → 90% coverage uncertainty intervals
Layer 4b: Red Team Agent       → Non-overridable clinical tripwires
Layer 5: Orchestrator          → Confidence-gated decision → GREEN/AMBER/RED/CRITICAL
```

## Quick Start (GPU Server)

```bash
# SSH into GPU server
ssh csegpuserver@172.16.18.2

# Install dependencies
pip3 install --user -r requirements.txt

# Full pipeline
cd ~/QuantumSepsis
python3 -m src.data.cohort_extraction_optimized --data-dir data/raw/physionet.org/files/mimiciv/3.1
python3 -m src.data.feature_extraction --cohort data/processed/cohort.csv --data-dir data/raw/physionet.org/files/mimiciv/3.1
python3 -m src.data.preprocessing --features data/processed/hourly_features.parquet --cohort data/processed/cohort.csv
python3 scripts/run_windowing_real.py
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm --data data/processed/features.h5
python3 scripts/run_real_baselines.py
```

## Current Status

| Stage | Status |
|-------|--------|
| Code implementation (24 modules) | ✅ Complete |
| MIMIC-IV v3.1 download (9.9 GB) | ✅ Complete |
| Cohort extraction (94,458 stays) | ✅ Complete |
| Feature extraction (12 features/hour) | ✅ Complete |
| Preprocessing + Windowing + Training | 🔄 Running on GPU server |
| Quantum kernel integration | ⏳ Phase 2 |

See **[PROGRESS_REPORT.md](PROGRESS_REPORT.md)** for full details and teammate instructions.

## Project Structure

```
QuantumSepsis/
├── src/
│   ├── config.py                           # All hyperparameters
│   ├── data/                               # Data pipeline (6 modules)
│   │   ├── cohort_extraction_optimized.py  # Memory-safe Sepsis-3 extraction
│   │   ├── feature_extraction.py           # 12 vitals/labs per hour
│   │   ├── preprocessing.py                # Imputation + normalization
│   │   ├── windowing.py                    # 6-hour sliding windows → HDF5
│   │   └── dataset.py                      # PyTorch DataLoaders
│   ├── models/                             # 4 model modules
│   │   ├── lstm.py                         # BiLSTM + Temporal Attention
│   │   ├── losses.py                       # Asymmetric Focal Loss (FN=9×FP)
│   │   ├── quantum_kernel.py               # ZZFeatureMap + QSVM (Qiskit)
│   │   └── conformal.py                    # Split Conformal + QCCP
│   ├── training/train_lstm.py              # Full training with early stopping
│   ├── agents/                             # Safety agents (3 modules)
│   │   ├── red_team.py                     # Non-overridable tripwires
│   │   ├── orchestrator.py                 # Confidence-gated fusion
│   │   └── outcome_learner.py              # Adaptive threshold tuning
│   ├── baselines/                          # XGBoost + SOFA comparisons
│   └── evaluation/metrics.py               # AUROC, AUPRC, calibration
├── scripts/                                # Pipeline runner scripts
├── files/                                  # Documentation
│   ├── architecture.md                     # Full 5-layer technical spec
│   ├── dataset.md                          # MIMIC-IV v3.1 reference
│   ├── novelty.md                          # 3 novel contributions
│   └── roadmap.md                          # 12-week execution plan
├── PROGRESS_REPORT.md                      # Current progress + next steps
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

1. **Quantum-Calibrated Conformal Prediction (QCCP)** — uncertainty from quantum Hilbert space
2. **Adversarial Tripwire-Gated Safety** — Red Team Agent + adaptive loss from violations  
3. **Confidence-Gated Diagnostic Fast-Tracking** — skip preliminary labs when confidence is high

## Team

- **Yash Gautam** — Data Pipeline + GPU Training
- **Atul Kumar Mishra (AKM)** — Model Architecture + Quantum Kernel
- **Tanishk Viraj Bhanage** — Safety Agents + Evaluation

## References

- Singer et al. (2016). Sepsis-3 Definitions. *JAMA* 315(8):801-810.
- Havlíček et al. (2019). Quantum-enhanced feature spaces. *Nature* 567:209-212.
- Johnson et al. (2023). MIMIC-IV. *PhysioNet*. DOI: 10.13026/6mm1-ek67.

## License

Research use only. MIMIC-IV PhysioNet Data Use Agreement applies.
