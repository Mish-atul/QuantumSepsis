# QuantumSepsis Shield — Project Backlog

> **Last Updated:** April 29, 2026  
> **Status Key:** ✅ Done · 🔄 In Progress · ⏳ Pending · ❌ Blocked

---

## ✅ DONE — Completed Work

### Infrastructure
- [x] GitHub repo created: https://github.com/Mish-atul/QuantumSepsis
- [x] GPU server access: 2× NVIDIA A100-PCIE-40GB (CUDA 13.0)
- [x] Python environment: PyTorch 2.11.0+cu130, all deps via `pip3 install --user`
- [x] MIMIC-IV v3.1 downloaded (9.9 GB) to `~/QuantumSepsis/data/raw/physionet.org/files/mimiciv/3.1/`

### Code — All 24 Modules Written
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
- [x] `src/agents/orchestrator.py` — Confidence-gated fusion (WATCH/AMBER/CRITICAL)
- [x] `src/agents/outcome_learner.py` — Adaptive thresholds + near-miss detection
- [x] `src/baselines/xgboost_baseline.py` — XGBoost on flattened windows
- [x] `src/baselines/sofa_baseline.py` — SOFA threshold baseline
- [x] `src/evaluation/metrics.py` — AUROC, AUPRC, Sensitivity@95%Spec, F1
- [x] `scripts/run_windowing_real.py` — Real-data windowing runner
- [x] `scripts/run_real_baselines.py` — Metric comparison script
- [x] `scripts/run_pipeline_autonomous.sh` — Full pipeline shell orchestrator

### Real Data Pipeline (All ran on GPU server)
- [x] **Cohort Extraction** → `data/processed/cohort.csv` (94,458 stays, 12,972 sepsis = 13.7%)
- [x] **Feature Extraction** → `data/processed/hourly_features.parquet` (~56 MB, 12 features/hour)
- [x] **Preprocessing** → `train/val/test_features.parquet` + `normalization_stats.json`
- [x] **Windowing** → `data/processed/features.h5` (~4.09M train windows, shape: N×6×12)
- [x] **LSTM Training** → `checkpoints/lstm_best.pt` (val AUROC = 0.7601)
- [x] **Embedding Extraction** → `data/processed/lstm_embeddings.npz` (16-dim, all splits)
- [x] **Baseline Runs** → `data/processed/pipeline_results_real.json`

### Phase 1 Results
| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec |
|---|---|---|---|
| SOFA Threshold | 0.5869 | 0.0159 | — |
| LSTM | 0.7891 | 0.0519 | 0.2997 |
| XGBoost | **0.8038** | **0.0576** | — |

### Bug Fixes
- [x] OOM fix: chunked CSV loading for labevents (124M rows) and chartevents (330M rows)
- [x] NaN timestamp fix: skip ICU stays with missing `intime`/`outtime`
- [x] RBF gamma bug fix: store `rbf_gamma_` from training, reuse on prediction (was causing AUROC 0.44 → 0.79)

### Quantum Kernel (Phase 2 — Partial)
- [x] RBF kernel (proxy) trained and evaluated: **Test AUROC = 0.7879**
- [x] Hyperparameter tuning: C=0.1, gamma=0.01, CV AUROC=0.7713
- [x] Support vector ratio: 86.7% (1734/2000)
- [x] Balanced subsampling: 2000 samples (1000 pos + 1000 neg)
- [x] PCA 16→8 (99.25% explained variance)

### Documentation
- [x] `IMPLEMENTATION_PLAN.md` — Step-by-step execution plan
- [x] `PROGRESS_REPORT.md` — Progress + teammate handoff notes
- [x] `README.md` — Project overview
- [x] `files/architecture.md`, `dataset.md`, `novelty.md`, `roadmap.md`

---

## 🔄 IN PROGRESS

### Qiskit Quantum Kernel Run
- **Status:** Screen session `qs_quantum` was running on GPU server at last update (April 23)
- **Task:** Retrieve results from `data/processed/quantum_results.json` on the server
- **What to check:**
  ```bash
  ssh csegpuserver@172.16.18.2
  screen -r qs_quantum   # Check if still running
  cat ~/QuantumSepsis/data/processed/quantum_results.json  # Check results
  ```
- **Expected:** True quantum AUROC using ZZFeatureMap (8 qubits, reps=2)
- **If dead:** Re-run with:
  ```bash
  screen -S qs_quantum
  cd ~/QuantumSepsis
  python3 -m src.models.quantum_kernel \
      --embeddings data/processed/lstm_embeddings.npz \
      --output data/processed/quantum_results.json \
      --max-train-samples 2000
  ```

---

## ⏳ PENDING — To Be Implemented

### HIGH PRIORITY

#### 1. Conformal Calibration on Real Data
**What:** Run `ConformalSepsisPredictor` using actual LSTM risk scores from `lstm_embeddings.npz`.  
**Why blocked:** Nothing — this can be done NOW without quantum kernel.  
**Steps:**
1. Load LSTM model (`lstm_best.pt`) + features.h5
2. Run inference on val split → get `val_scores`, `val_labels`
3. Run inference on test split → get `test_scores`, `test_labels`
4. Call `predictor.calibrate(val_scores, val_labels)` → get `q_α`
5. Call `predictor.verify_coverage(test_scores, test_labels)` → verify ≥ 90% coverage
6. Record: `q_α`, `mean_width`, `pct_escalated`

**Expected output:**
- `q_α` value (how wide prediction intervals will be)
- Empirical coverage ≥ 0.90 (guaranteed by theory)
- Fraction of test windows that would trigger uncertainty escalation

**Script to write:**
```python
# scripts/run_conformal_calibration.py
from src.models.lstm import SepsisLSTM
from src.models.conformal import ConformalSepsisPredictor
import torch, h5py, numpy as np

model = SepsisLSTM()
model.load_state_dict(torch.load("checkpoints/lstm_best.pt"))
model.eval()

with h5py.File("data/processed/features.h5", "r") as f:
    X_val = torch.tensor(f["X_val"][:])
    y_val = f["y_val"][:]
    X_test = torch.tensor(f["X_test"][:])
    y_test = f["y_test"][:]

with torch.no_grad():
    val_scores = model(X_val)["risk_score"].numpy()
    test_scores = model(X_test)["risk_score"].numpy()

predictor = ConformalSepsisPredictor()
predictor.calibrate(val_scores, y_val)
stats = predictor.verify_coverage(test_scores, y_test)
print(stats)
```

---

#### 2. End-to-End Orchestrator Validation
**What:** Run the full pipeline (LSTM → Conformal → RedTeam → Orchestrator) on the real test set.  
**Why blocked:** Needs conformal calibration first (item #1 above).  
**Dependency:** Conformal calibration must be done first.

**Steps:**
1. Complete conformal calibration → `calibrated_predictor`
2. Load `normalization_stats.json` → initialize `RedTeamAgent(use_normalized=True)`
3. For each test window:
   - LSTM → `risk_score`
   - Conformal → `lower, upper`
   - RedTeam → `assessment`
   - Orchestrator → `decision`
4. Aggregate: distribution of WATCH/AMBER/CRITICAL decisions, fast-track rate
5. Compare alert distribution with ground truth sepsis labels

**Script to write:** `scripts/run_e2e_validation.py`

**Expected outputs:**
- Alert distribution across test set (% WATCH, AMBER, CRITICAL, FAST-TRACK)
- Sensitivity/specificity of CRITICAL alerts vs. sepsis labels
- False negative rate at WATCH level (missed sepsis cases)
- Red Team override rate (% of cases where tripwires fired)

---

#### 3. LSTM Hyperparameter Tuning
**What:** XGBoost (0.8038) is beating LSTM (0.7891). Close the gap.  
**Why:** LSTM should outperform XGBoost on sequential data if tuned properly.

**Things to try (in order of effort):**
- Increase hidden_dim: 128 → 256
- Add a third LSTM layer (`n_layers: 3`)
- Increase embedding_dim: 16 → 32 (also update quantum PCA input)
- Reduce prediction_horizon_hours: 4 → 2 (easier task, higher AUROC)
- Add class-balanced sampling in DataLoader
- Try SMOTE on training windows

**Run:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm \
    --data data/processed/features.h5
```

---

#### 4. AUPRC / Class Imbalance Investigation
**What:** AUPRC is ~0.05 across ALL models (random baseline is 0.014 for 13.7% prevalence).  
**Problem:** Sliding window approach creates massive class imbalance in windows.

**Root cause analysis:**
- 94,458 stays → ~4.09M windows
- Sepsis-positive windows = only ~4-6hr before onset per patient
- Most ICU stays are long → vast majority of windows are negative

**Options to investigate:**
- Check actual positive rate in windows: `h5py.File → y_train.mean()`
- Try focal loss with higher γ (3.0 or 4.0)
- Try under-sampling negative windows during training
- Try shorter prediction horizon (2hr instead of 4hr)
- Report window-level vs. stay-level metrics separately

---

### MEDIUM PRIORITY

#### 5. QCCP Integration (after quantum kernel results)
**What:** Run `QuantumCalibratedConformal` using quantum kernel centroids.  
**Blocked by:** Qiskit quantum kernel results (item in progress).  
**Steps:**
1. Fit quantum kernel on training embeddings
2. Compute sepsis centroids: `quantum.get_centroids(X_train_pca, y_train_pca, n=5)`
3. Initialize `QuantumCalibratedConformal`
4. `predictor.set_quantum_kernel(kernel_fn, centroids)`
5. `predictor.calibrate_quantum(val_embeddings, val_labels)`
6. Compare QCCP interval widths vs. standard conformal widths
7. Verify coverage ≥ 90%

**Expected:** Tighter intervals than standard conformal (key novelty claim N1).

---

#### 6. Quantum vs. Classical Comparison Table
**What:** Final comprehensive results table for paper/report.  
**Blocked by:** Qiskit quantum kernel results.

**Target table:**
| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec | Conformal Width |
|---|---|---|---|---|
| SOFA Threshold | 0.5869 | 0.0159 | — | — |
| XGBoost | 0.8038 | 0.0576 | — | — |
| LSTM | 0.7891 | 0.0519 | 0.2997 | TBD |
| RBF Quantum (Fixed) | 0.7879 | 0.0520 | 0.2999 | TBD |
| **Qiskit Quantum** | **🔄** | **🔄** | **🔄** | **TBD** |
| **QCCP Conformal** | — | — | — | **TBD** |

---

#### 7. Outcome Learning Agent — Live Simulation
**What:** Simulate the OutcomeLearningAgent on the test set to validate adaptive thresholds.  
**Why:** The agent code is complete but never run on real MIMIC-IV outcome data.

**Steps:**
1. For each test case: use ground truth `sepsis_label` as `actual_sepsis`
2. Feed orchestrator decisions + true labels into `agent.record_outcome()`
3. Observe: threshold drift, near-miss detection, loss multiplier escalation
4. Report: per-unit sensitivity before vs. after threshold adaptation

---

### LOW PRIORITY

#### 8. IBM Quantum Hardware Run (Optional / Future)
**What:** Replace AerSimulator with real IBM Quantum backend.  
**Config:** `quantum.backend = "ibm_quantum"`, `quantum.ibm_backend = "ibm_brisbane"`  
**Blocked by:** IBM Quantum account + token (`quantum.ibm_token`).  
**Note:** Likely impractical for current deadline. Simulator results are sufficient for paper.

#### 9. W&B Dashboard Setup
**What:** Ensure W&B logging is active during LSTM training runs.  
**Config:** `training.use_wandb = True`, `training.wandb_project = "quantumsepsis-shield"`  
**Note:** Currently may be disabled if W&B not installed on GPU server.

#### 10. Attention Weight Visualization
**What:** Visualize which hours in the 6-hour window the LSTM attends to most for sepsis cases.  
**How:** Use `model(x, return_attention=True)["attention_weights"]` on positive test cases.  
**Value:** Interpretability figure for paper.

#### 11. Final Paper / Report Figures
- Precision-Recall curve (all models overlaid)
- ROC curve (all models overlaid)
- Conformal prediction interval visualization (width histogram)
- Red Team tripwire frequency chart
- Attention weight heatmap for sample sepsis patients

---

## Summary Table

| # | Task | Priority | Can Start Now? | Blocked By | Est. Time |
|---|---|---|---|---|---|
| 1 | Conformal calibration on real data | 🔴 High | ✅ Yes | Nothing | 1-2 hrs |
| 2 | End-to-end orchestrator validation | 🔴 High | ✅ Yes (after #1) | Task #1 | 2-3 hrs |
| 3 | LSTM hyperparameter tuning | 🔴 High | ✅ Yes | Nothing | 2-4 hrs |
| 4 | AUPRC / class imbalance investigation | 🔴 High | ✅ Yes | Nothing | 2-4 hrs |
| QK | Retrieve Qiskit quantum kernel results | 🔴 High | ✅ Yes | Server access | 30 min |
| 5 | QCCP integration | 🟡 Medium | ❌ No | Qiskit results | 2-3 hrs |
| 6 | Final comparison table | 🟡 Medium | ❌ No | Qiskit results | 1 hr |
| 7 | Outcome learner simulation | 🟡 Medium | ✅ Yes | Nothing | 1-2 hrs |
| 8 | IBM Quantum hardware run | 🟢 Low | ❌ No | IBM account | Unknown |
| 9 | W&B dashboard | 🟢 Low | ✅ Yes | Nothing | 30 min |
| 10 | Attention visualization | 🟢 Low | ✅ Yes | Nothing | 1 hr |
| 11 | Paper figures | 🟢 Low | ✅ Partial | Results | 2-3 hrs |

---

## Recommended Execution Order

```
Day 1:
  1. SSH to server → check Qiskit run status (qs_quantum screen session)
  2. Run conformal calibration script on LSTM scores
  3. Investigate AUPRC / positive rate in windows

Day 2:
  4. Run end-to-end orchestrator on test set
  5. Try LSTM hyperparameter tuning (hidden_dim=256)
  6. Simulate outcome learner on test set

Day 3 (after quantum results):
  7. Run QCCP integration
  8. Compile final comparison table
  9. Generate paper figures
```
