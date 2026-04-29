# QuantumSepsis Shield — Project Backlog

> **Last Updated:** April 30, 2026  
> **Status Key:** ✅ Done · 🔄 In Progress · ⏳ Pending · ❌ Blocked

---

## ✅ DONE — Completed Work

### Infrastructure
- [x] GitHub repo created: https://github.com/Mish-atul/QuantumSepsis
- [x] GPU server access: 2× NVIDIA A100-PCIE-40GB (CUDA 13.0)
- [x] Python environment: PyTorch 2.11.0+cu130, all deps via `pip3 install --user`
- [x] MIMIC-IV v3.1 downloaded (9.9 GB) to `~/QuantumSepsis/data/raw/physionet.org/files/mimiciv/3.1/`

### Code — All 24 Source Modules Written
- [x] `src/config.py` — Dataclass config for all components
- [x] `src/data/cohort_extraction.py` — Original (broken on real data, OOM)
- [x] `src/data/cohort_extraction_optimized.py` — Chunked loading fix ✅ Memory safe
- [x] `src/data/feature_extraction.py` — 12 features per hour
- [x] `src/data/preprocessing.py` — Forward-fill, median imputation, z-score, split
- [x] `src/data/windowing.py` — 6-hour sliding windows → HDF5
- [x] `src/data/dataset.py` — PyTorch DataLoaders
- [x] `src/models/lstm.py` — BiLSTM + TemporalAttention + 16-dim embedding
- [x] `src/models/losses.py` — AsymmetricFocalLoss (FN:FP = 9:1)
- [x] `src/models/quantum_kernel.py` — ZZFeatureMap QSVM + RBF fallback
- [x] `src/models/conformal.py` — Split conformal + QCCP
- [x] `src/training/train_lstm.py` — Full training loop (AdamW, cosine LR, early stopping, W&B)
- [x] `src/agents/red_team.py` — 5 clinical tripwires, non-overridable
- [x] `src/agents/orchestrator.py` — Confidence-gated fusion (WATCH/AMBER/CRITICAL/FAST-TRACK)
- [x] `src/agents/outcome_learner.py` — Adaptive thresholds + near-miss detection
- [x] `src/baselines/xgboost_baseline.py` — XGBoost on flattened windows
- [x] `src/baselines/sofa_baseline.py` — SOFA threshold baseline
- [x] `src/evaluation/metrics.py` — AUROC, AUPRC, Sensitivity@95%Spec, F1

### Pipeline Scripts — All Written
- [x] `scripts/run_windowing_real.py` — Real-data windowing runner
- [x] `scripts/run_real_baselines.py` — Metric comparison script
- [x] `scripts/run_pipeline_autonomous.sh` — Full pipeline shell orchestrator
- [x] `scripts/run_conformal_calibration.py` — LSTM → q_alpha → conformal intervals ✅ **NEW**
- [x] `scripts/run_e2e_validation.py` — LSTM + Conformal + RedTeam + Orchestrator fusion ✅ **NEW**
- [x] `scripts/run_outcome_learning_simulation.py` — Adaptive feedback loop on decisions ✅ **NEW**
- [x] `scripts/analyze_class_imbalance.py` — AUPRC investigation + recommendations ✅ **NEW**
- [x] `scripts/run_lstm_tuning.py` — 5 hyperparameter experiments to beat XGBoost ✅ **NEW**

### Tests — 31 Edge Case Tests Written
- [x] `tests/test_conformal_calibration.py` — 14 tests ✅ **NEW**
  - Coverage guarantee, boundary clipping, extreme label distributions (0%/100%)
  - Single sample calibration, batch size consistency, full pipeline with tempfiles
- [x] `tests/test_e2e_validation.py` — 17 tests ✅ **NEW**
  - Normal run, missing files (checkpoint/HDF5/calibration JSON)
  - All-zero/all-one labels, wide/zero q_alpha effects
  - Batch size determinism, norm stats with/without, confidence ranges

### Real Data Pipeline (All ran on GPU server)
- [x] **Cohort Extraction** → `data/processed/cohort.csv` (94,458 stays, 12,972 sepsis = 13.7%)
- [x] **Feature Extraction** → `data/processed/hourly_features.parquet` (~56 MB, 12 features/hour)
- [x] **Preprocessing** → `train/val/test_features.parquet` + `normalization_stats.json`
- [x] **Windowing** → `data/processed/features.h5` (~4.09M train windows, shape: N×6×12)
- [x] **LSTM Training** → `checkpoints/lstm_best.pt` (val AUROC = 0.7601)
- [x] **Embedding Extraction** → `data/processed/lstm_embeddings.npz` (16-dim, all splits)
- [x] **Baseline Runs** → `data/processed/pipeline_results_real.json`

### Phase 3 Integration Results
- [x] **Conformal calibration** → `data/processed/conformal_calibration.json` (q_alpha = 0.3923)
- [x] **E2E validation** → `data/processed/e2e_validation_results.json` (796,893 windows)
- [x] **Outcome learning simulation** → `data/processed/outcome_learning_results.json` (FN=0, near-misses=2,155)
- [x] **Stay-level metrics** → `data/processed/stay_level_metrics.json` (AUROC 0.8618, AUPRC 0.5012)
- [x] **LSTM tuning (exp5_combined)** attempted — failed to beat baseline (val AUROC 0.7583)

### Phase 1 Results
| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec |
|---|---|---|---|
| SOFA Threshold | 0.5869 | 0.0159 | — |
| LSTM | 0.7891 | 0.0519 | 0.2997 |
| XGBoost | **0.8038** | **0.0576** | — |
| RBF Quantum Kernel | 0.7879 | 0.0520 | 0.2999 |

### Bug Fixes
- [x] OOM fix: chunked CSV loading for labevents (124M rows) and chartevents (330M rows)
- [x] NaN timestamp fix: skip ICU stays with missing `intime`/`outtime`
- [x] RBF gamma bug fix: store `rbf_gamma_` from training, reuse on prediction (AUROC 0.44 → 0.79)

### Quantum Kernel (Phase 2 — Partial)
- [x] RBF kernel (proxy) trained and evaluated: **Test AUROC = 0.7879**
- [x] Hyperparameter tuning: C=0.1, gamma=0.01, CV AUROC=0.7713
- [x] Support vector ratio: 86.7% (1734/2000)
- [x] Balanced subsampling: 2000 samples (1000 pos + 1000 neg)
- [x] PCA 16→8 (99.25% explained variance)

### Documentation
- [x] `agents.md` — Complete project knowledge base (end-to-end reference)
- [x] `backlog.md` — Done vs pending task tracker
- [x] `IMPLEMENTATION_PLAN.md` — Step-by-step execution plan with credentials
- [x] `PROGRESS_REPORT.md` — Progress + teammate handoff notes
- [x] `README.md` — Project overview with quick start guide
- [x] `files/architecture.md`, `dataset.md`, `novelty.md`, `roadmap.md`

---

## ❌ BLOCKED

### Qiskit Quantum Kernel Run
- **Status:** Qiskit kernel crashed / infeasible on server
- **Fallback:** RBF quantum kernel validated (AUROC 0.7879)
- **Action:** Keep RBF results as quantum validation baseline

---

## ⏳ PENDING — To Be Implemented

### HIGH PRIORITY

#### 1. Run Phase 2 Scripts on GPU Server
**Status:** ✅ Completed (conformal calibration, E2E validation, outcome learning, class imbalance analysis)

---

#### 2. LSTM Hyperparameter Tuning (Close AUROC Gap)
**Status:** ✅ Attempted; exp5_combined underperformed baseline (val AUROC 0.7583)

---

#### 3. Stay-Level Metrics
**Status:** ✅ Completed (stay-level AUROC 0.8618, AUPRC 0.5012)

---

#### 4. Additional Features (12 → 25+ for AUROC boost)
**What:** Add more lab features from MIMIC-IV to improve model discriminative power.  
**Missing high-value features:**
- BUN (labevents 51006) — renal function
- Bilirubin (labevents 50885) — liver SOFA
- INR/PT (labevents 51237) — coagulation
- Procalcitonin (labevents 50911) — infection biomarker
- FiO2 (chartevents 223835) — respiratory support
- Urine output (outputevents) — renal SOFA
- Age, gender (patients) — demographics

---

### MEDIUM PRIORITY

#### 5. QCCP Integration (after quantum kernel results)
**What:** Run `QuantumCalibratedConformal` using quantum kernel centroids.  
**Blocked by:** Qiskit quantum kernel results.  
**Steps:**
1. Fit quantum kernel on training embeddings
2. Compute sepsis centroids: `quantum.get_centroids(X_train_pca, y_train_pca, n=5)`
3. Initialize `QuantumCalibratedConformal`
4. Compare QCCP interval widths vs. standard conformal widths
5. Verify coverage ≥ 90%

---

#### 6. Visualization Script
**Status:** ✅ Implemented (`scripts/generate_figures.py`, outputs in `figures/`)

---

#### 7. Threshold Auto-Loading (Close Feedback Loop)
**What:** Make Orchestrator read updated thresholds from `outcome_learning_results.json`.  
**Why:** Currently the OutcomeLearningAgent computes new thresholds but the Orchestrator uses hardcoded defaults. This would fully close the feedback loop.

---

#### 8. Class-Balanced DataLoader
**What:** Implement undersampling (10:1 neg:pos ratio) inside `dataset.py`.  
**Why:** Addresses the root cause of low AUPRC (~0.05).

---

### LOW PRIORITY

#### 9. IBM Quantum Hardware Run (Optional)
**What:** Replace AerSimulator with real IBM Quantum backend.  
**Note:** Likely impractical for current deadline. Simulator results are sufficient.

#### 10. W&B Dashboard Setup
**What:** Ensure W&B logging is active during LSTM training runs.

#### 11. Attention Weight Visualization
**What:** Visualize which hours in the 6-hour window the LSTM attends to most.

---

## Summary Table

| # | Task | Priority | Status | Blocked By | Est. Time |
|---|---|---|---|---|---|
| 1 | Run Phase 2 scripts on server | 🔴 High | ✅ Done | — | — |
| 2 | LSTM tuning (5 experiments) | 🔴 High | ✅ Done (no gain) | — | — |
| 3 | Stay-level metrics | 🔴 High | ✅ Done | — | — |
| 4 | Add more features (12→25+) | 🔴 High | ⏳ Not started | Server re-run | 4-6 hrs |
| QK | Retrieve Qiskit quantum results | 🔴 High | 🔄 In progress | Server access | 30 min |
| 5 | QCCP integration | 🟡 Medium | ❌ Blocked | Qiskit results | 2-3 hrs |
| 6 | Visualization script | 🟡 Medium | ✅ Done | — | — |
| 7 | Threshold auto-loading | 🟡 Medium | ⏳ Not started | Nothing | 1 hr |
| 8 | Class-balanced DataLoader | 🟡 Medium | ⏳ Not started | Nothing | 2 hrs |
| 9 | IBM Quantum hardware | 🟢 Low | ⏳ Not started | IBM account | Unknown |
| 10 | W&B dashboard | 🟢 Low | ⏳ Not started | Nothing | 30 min |
| 11 | Attention visualization | 🟢 Low | ⏳ Not started | Nothing | 1 hr |

---

## Recommended Next Steps

```
Immediate (today):
  1. SSH to server → check Qiskit run status
  2. Run all Phase 2 scripts on real data
  3. Run LSTM tuning experiment exp5_combined

Next session:
  4. Implement stay-level metrics
  5. Start visualization script for paper figures
  6. Close feedback loop (threshold auto-loading)

After quantum results:
  7. QCCP integration
  8. Final comparison table
  9. Paper figures with all models
```
