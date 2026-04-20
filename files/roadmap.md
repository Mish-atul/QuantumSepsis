# QuantumSepsis Shield — 12-Week Execution Roadmap

> **Team:** Yash Gautam (YG), Atul Kumar Mishra (AKM), Tanishk Viraj Bhanage (TVB)

---

## Phase 1: Foundation & Classical Baseline (Weeks 1–4)

### Week 1: Project Setup & Data Access
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Set up project repo, directory structure, CI | AKM | Git repo with modular structure | No |
| PhysioNet credential verification + MIMIC-IV download start | YG | Download script running on GPU server | No |
| Literature review: quantum kernels + conformal prediction | TVB | 2-page summary in `files/literature.md` | No |
| Create environment: Python 3.10, PyTorch 2.x, Qiskit 1.x | AKM | `requirements.txt` + conda env | No |
| Architecture documentation finalization | All | `files/architecture.md` complete | No |

**MVC Checkpoint:** Environment set up, MIMIC-IV download initiated, architecture doc complete.

---

### Week 2: Data Pipeline — Cohort Extraction
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Write Sepsis-3 cohort extraction SQL/Python | YG | `src/data/cohort_extraction.py` | No |
| SOFA score computation module | AKM | `src/data/sofa.py` | No |
| Suspected infection detection module | TVB | `src/data/infection_detection.py` | No |
| Run cohort extraction on MIMIC-IV | YG | `data/processed/cohort.csv` | No |
| Verify cohort statistics (expected: 22K-28K sepsis stays) | All | Validation report | No |

**MVC Checkpoint:** Sepsis cohort extracted with correct prevalence (~25%).

---

### Week 3: Feature Engineering & Windowing
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Vital signs extraction (chartevents) | YG | `src/data/vitals_extraction.py` | No |
| Lab values extraction (labevents) | AKM | `src/data/labs_extraction.py` | No |
| 1-hour binning + forward fill + median imputation | TVB | `src/data/preprocessing.py` | No |
| 6-hour sliding window tensor generation | AKM | `src/data/windowing.py` | No |
| Train/val/test temporal split | YG | Split manifests in `data/processed/` | No |
| Z-score normalization (train stats only) | TVB | `src/data/normalization.py` | No |
| Save to HDF5 format | AKM | `data/processed/features.h5` | No |

**MVC Checkpoint:** Feature tensor (N, 6, 12) in HDF5, verified with basic statistics.

---

### Week 4: LSTM Baseline Training
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| `SepsisLSTM` model class (PyTorch) | AKM | `src/models/lstm.py` | No |
| Asymmetric focal loss implementation | TVB | `src/models/losses.py` | No |
| Training loop + W&B logging | YG | `src/training/train_lstm.py` | **Yes** |
| Evaluation metrics (AUROC, AUPRC, sensitivity@95spec) | AKM | `src/evaluation/metrics.py` | No |
| Classical LSTM training on GPU cluster | YG + AKM | Trained model checkpoint | **Yes** |
| XGBoost baseline comparison | TVB | `src/baselines/xgboost_baseline.py` | No |
| SOFA score threshold baseline | TVB | `src/baselines/sofa_baseline.py` | No |

**MVC Checkpoint:** LSTM AUROC ≥ 0.80 on validation set. XGBoost and SOFA baselines recorded.

> ⚠️ **Internal Deadline:** Week 4 = 2 weeks before Phase 1 mentor review (Week 6)

---

## Phase 2: Quantum Integration & Safety Layers (Weeks 5–8)

### Week 5: Quantum Kernel Module
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| PCA reduction: 16→8 dimensions | TVB | `src/models/pca_reducer.py` | No |
| `QuantumKernelSepsis` class (Qiskit) | AKM | `src/models/quantum_kernel.py` | No |
| ZZFeatureMap circuit (8 qubits, reps=2) | AKM | Circuit diagram + code | No |
| Kernel matrix computation (AerSimulator) | AKM | Kernel matrix on training set | No |
| QSVM with precomputed kernel (sklearn) | YG | `src/models/qsvm.py` | No |

**MVC Checkpoint:** Quantum kernel matrix computed, QSVM trained on LSTM embeddings.

---

### Week 6: Conformal Prediction + QCCP
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Classical conformal prediction wrapper (MAPIE) | TVB | `src/models/conformal.py` | No |
| Quantum-calibrated conformal (QCCP) — Novelty 1 | AKM + TVB | `src/models/qccp.py` | No |
| Calibration set creation + coverage verification | YG | Coverage ≥ 90% verified | No |
| Uncertainty width analysis | TVB | Width vs. coverage plots | No |

**Phase 1 Mentor Review (end of Week 6)**

---

### Week 7: Red Team Agent
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| `RedTeamAgent` class implementation | YG | `src/agents/red_team.py` | No |
| Tripwire threshold calibration on MIMIC-IV | AKM | Threshold sensitivity analysis | No |
| Non-overridability testing | TVB | Unit tests for override prevention | No |
| Adaptive loss feedback mechanism — Novelty 2 | AKM | `src/agents/adaptive_loss.py` | No |

**MVC Checkpoint:** Red Team Agent triggers correctly on known sepsis cases.

---

### Week 8: Full Pipeline Integration
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| `ConfidenceGatedOrchestrator` — Novelty 3 | YG | `src/agents/orchestrator.py` | No |
| End-to-end pipeline wiring (Layers 1–5) | All | `src/pipeline/pipeline.py` | No |
| Pipeline simulation on MIMIC-IV test set | AKM | End-to-end metrics | **Yes** |
| Outcome Learning Agent (threshold adaptation) | TVB | `src/agents/outcome_learner.py` | No |

**MVC Checkpoint:** Full 5-layer pipeline runs end-to-end with AUROC ≥ 0.85.

> ⚠️ **Internal Deadline:** Week 8 = 2 weeks before Phase 2 mentor review (Week 10)

---

## Phase 3: Validation, Ablation & Presentation (Weeks 9–12)

### Week 9: Ablation Studies & Baseline Comparisons
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Full baseline comparison table | All | `files/baseline_comparison.md` | **Yes** |
| Ablation: remove quantum kernel → measure drop | AKM | Ablation results | **Yes** |
| Ablation: remove Red Team Agent → measure FN increase | YG | Ablation results | No |
| Ablation: remove conformal prediction → measure calibration | TVB | Ablation results | No |
| Lead time analysis (hours before onset) | YG | Lead time distribution plots | No |

| Baseline | Expected AUROC | Our System Target |
|----------|---------------|-------------------|
| SOFA threshold | 0.65–0.70 | — |
| NEWS2 score | 0.70–0.75 | — |
| XGBoost (raw features) | 0.78–0.82 | — |
| Classical LSTM only | 0.80–0.84 | — |
| **QuantumSepsis Shield** | — | **≥ 0.85** |

---

### Week 10: IBM Quantum Cloud Validation
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Submit quantum circuits to IBM Quantum Runtime | AKM | Real hardware kernel matrix | No |
| Compare AerSimulator vs. real hardware results | AKM | Noise impact analysis | No |
| Error mitigation strategies | TVB | Mitigated vs. raw comparison | No |
| Indian ICU adaptation roadmap document | YG | `files/indian_icu_roadmap.md` | No |

**Phase 2 Mentor Review (end of Week 10)**

---

### Week 11: Documentation & Presentation Prep
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Final technical report | All | `files/final_report.md` | No |
| Presentation slides (2-min + full versions) | YG | Slide deck | No |
| Demo video recording | AKM | 3-min demo video | No |
| Practice presentations (each member speaks to all parts) | All | Practice sessions | No |

---

### Week 12: Polish & Final Submission
| Task | Owner | Deliverable | GPU? |
|------|-------|-------------|------|
| Code cleanup + documentation | AKM | Clean codebase | No |
| Final model checkpoint + reproducibility verification | TVB | Seed verification | No |
| Final presentation rehearsal | All | Timed rehearsal | No |
| Submission | YG | Final submission package | No |

> ⚠️ **Internal Deadline:** Week 12 = final submission

---

## GPU Cluster Usage Plan

| Week | GPU Usage | Duration (est.) | Task |
|------|-----------|-----------------|------|
| 4 | Heavy | 8–12 hours | LSTM training (full MIMIC-IV) |
| 5 | Light | 2–4 hours | Quantum kernel simulation |
| 8 | Medium | 4–6 hours | Full pipeline evaluation |
| 9 | Heavy | 8–12 hours | Ablation studies (multiple runs) |

**Total estimated GPU hours:** 30–40 hours across 12 weeks

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MIMIC-IV download fails | Low | High | Use BigQuery as fallback |
| GPU cluster unavailable | Medium | High | Pre-train locally on subset; scale up when available |
| Quantum kernel too slow | Medium | Medium | Reduce to 4 qubits; use PCA to 4 dims |
| AUROC < 0.85 target | Medium | High | Hyperparameter sweep; ensemble methods |
| Team member unavailable | Low | Medium | Cross-training (all members know all parts) |
