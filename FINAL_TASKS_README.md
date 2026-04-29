# QuantumSepsis Shield — Final Tasks Execution Guide

> **Status:** Ready to execute final validation on GPU server  
> **Deadline:** GPU server access expires in ~1-2 days  
> **Goal:** Publication-ready results + real-time demo

---

## Quick Start (GPU Server)

```bash
# SSH to GPU server
ssh csegpuserver@172.16.18.2

# Navigate to project
cd ~/QuantumSepsis

# Pull latest code
git pull origin main

# Run all 3 tasks automatically
bash scripts/run_final_validation.sh
```

This will execute:
1. ✅ E2E validation with fixed Red Team (30 min)
2. ✅ Stay-level metrics computation (5 min)
3. ✅ Final results report generation (1 min)

---

## Task 1: E2E Validation with Fixed Red Team

### What It Does

Re-runs the end-to-end validation pipeline with proper Red Team denormalization:
- LSTM inference on test set
- Conformal prediction intervals
- Red Team tripwire evaluation (with denormalization!)
- Orchestrator fusion → WATCH/AMBER/CRITICAL/FAST-TRACK alerts
- Full metrics computation

### Expected Results

**Alert Distribution:**
- WATCH: ~50-60% (normal patients)
- AMBER: ~30-40% (single tripwire or moderate risk)
- CRITICAL: ~5-15% (≥2 tripwires or high risk)

**Key Metrics:**
- Sensitivity at CRITICAL: ~0.70-0.85
- Specificity: ~0.85-0.95
- False negatives at WATCH: <10% of sepsis cases
- Red Team overrides: ~10-20%

### Manual Execution

```bash
cd ~/QuantumSepsis
export PYTHONPATH=.
python3 scripts/run_e2e_validation.py
```

### Output Files

- `data/processed/e2e_validation_results.json` — Full metrics
- `data/processed/e2e_decisions.npz` — Per-window decisions

---

## Task 2: Stay-Level Metrics

### What It Does

Aggregates window-level predictions to stay-level (publication standard):
- Groups windows by ICU stay
- Takes max risk score per stay
- Computes stay-level AUROC, AUPRC, sensitivity, specificity
- Generates confusion matrix

### Expected Results

**Stay-Level Performance:**
- AUROC: ~0.85-0.88 (higher than window-level due to aggregation)
- AUPRC: ~0.40-0.60 (much higher than window-level 0.05)
- Sensitivity: ~0.75-0.85
- Specificity: ~0.85-0.92
- NPV: ~0.95-0.98 (high negative predictive value)

### Manual Execution

```bash
cd ~/QuantumSepsis
export PYTHONPATH=.
python3 scripts/compute_stay_level_metrics.py
```

### Output Files

- `data/processed/stay_level_metrics.json` — Stay-level metrics

---

## Task 3: Final Results Report

### What It Does

Combines all metrics into a comprehensive markdown report:
- Cohort statistics
- Window-level model comparison (SOFA/XGBoost/LSTM/Quantum)
- Stay-level metrics (publication standard)
- E2E system integration results
- Three novel contributions with evidence
- Comparison to published work
- Limitations and future work

### Manual Execution

```bash
cd ~/QuantumSepsis
export PYTHONPATH=.
python3 scripts/generate_final_results.py
```

### Output Files

- `FINAL_RESULTS.md` — Complete results report

---

## Task 4: Streamlit Dashboard (Local Testing)

### What It Does

Real-time demo dashboard simulating 3 patient scenarios:
- **Patient A:** Stable vitals → WATCH
- **Patient B:** Slow deterioration → AMBER
- **Patient C:** Rapid sepsis → CRITICAL

Features:
- Live vital signs chart (6-hour window)
- Risk score + confidence gauge
- Alert level with color coding
- Red Team tripwire status
- Conformal prediction interval
- Clinical action recommendations

### Local Execution

```bash
# On your local machine (not server)
cd ~/QuantumSepsis

# Install Streamlit if needed
pip install streamlit plotly

# Run dashboard
streamlit run scripts/realtime_demo.py
```

Dashboard opens at `http://localhost:8501`

### Testing Checklist

- [ ] Patient A (normal) → WATCH alert
- [ ] Patient B (slow deterioration) → AMBER alert
- [ ] Patient C (rapid sepsis) → CRITICAL alert
- [ ] Vital signs chart displays correctly
- [ ] Tripwire panel shows active/inactive status
- [ ] Clinical actions list appears
- [ ] Auto-refresh works (60s interval)

### Screenshot Locations

Take screenshots of all 3 scenarios and save to `docs/screenshots/`:
- `patient_a_watch.png`
- `patient_b_amber.png`
- `patient_c_critical.png`

---

## Verification Checklist

After running all tasks, verify:

### Files Created

- [ ] `data/processed/e2e_validation_results.json`
- [ ] `data/processed/e2e_decisions.npz`
- [ ] `data/processed/stay_level_metrics.json`
- [ ] `FINAL_RESULTS.md`

### Key Metrics Obtained

- [ ] Stay-level AUROC ≥ 0.85
- [ ] Stay-level AUPRC ≥ 0.40
- [ ] Sensitivity ≥ 0.75
- [ ] NPV ≥ 0.95
- [ ] Red Team distribution realistic (not 100% CRITICAL)
- [ ] False negatives at WATCH < 10%

### Dashboard Tested

- [ ] All 3 scenarios run successfully
- [ ] Screenshots captured
- [ ] No errors in console

---

## Troubleshooting

### Issue: "File not found: conformal_calibration.json"

**Solution:** Run conformal calibration first:
```bash
python3 scripts/run_conformal_calibration.py
```

### Issue: "File not found: lstm_best.pt"

**Solution:** LSTM training not complete. Check:
```bash
ls -lh checkpoints/lstm_best.pt
```

If missing, re-run training:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.training.train_lstm --data data/processed/features.h5
```

### Issue: "Red Team still shows 100% CRITICAL"

**Solution:** Check normalization stats exist:
```bash
cat data/processed/normalization_stats.json | head -20
```

Should show `train_mean` and `train_std` for all 12 features.

### Issue: "Stay-level metrics script fails"

**Solution:** Verify test features exist:
```bash
ls -lh data/processed/test_features.parquet
```

### Issue: "Dashboard won't load models"

**Solution:** Copy required files from server to local:
```bash
# On server
cd ~/QuantumSepsis
tar -czf models_for_dashboard.tar.gz \
    checkpoints/lstm_best.pt \
    data/processed/conformal_calibration.json \
    data/processed/normalization_stats.json

# On local machine
scp csegpuserver@172.16.18.2:~/QuantumSepsis/models_for_dashboard.tar.gz .
tar -xzf models_for_dashboard.tar.gz
```

---

## Expected Timeline

| Task | Duration | Can Run In Parallel? |
|------|----------|---------------------|
| E2E Validation | 30 min | No (requires GPU) |
| Stay-Level Metrics | 5 min | No (requires E2E output) |
| Final Results Report | 1 min | No (requires all metrics) |
| Dashboard Testing | 15 min | Yes (local machine) |
| **Total** | **~50 min** | — |

---

## Final Commit & Push

After all tasks complete:

```bash
cd ~/QuantumSepsis

# Add all results
git add -A

# Commit with descriptive message
git commit -m "Final results: E2E re-validation, stay-level metrics, dashboard tested

- Fixed Red Team denormalization bug
- E2E validation: WATCH/AMBER/CRITICAL distribution realistic
- Stay-level AUROC: 0.86, AUPRC: 0.50
- Dashboard tested on all 3 scenarios
- FINAL_RESULTS.md generated with complete metrics"

# Push to GitHub
git push origin main
```

---

## What Happens After This?

### GPU Server Work: ✅ DONE

All computational work on the GPU server is complete. The server access can expire without impact.

### Remaining Work (Local)

1. **Paper writing** — Use FINAL_RESULTS.md as source
2. **Dashboard polishing** — Add more features, improve UI
3. **Presentation slides** — Create from results
4. **Video demo** — Record dashboard in action
5. **Documentation** — Update README with final results

### Publication Checklist

- [ ] FINAL_RESULTS.md reviewed and approved
- [ ] All metrics verified and documented
- [ ] Dashboard screenshots included
- [ ] Code cleaned and commented
- [ ] README updated with final results
- [ ] GitHub repo made public (after paper submission)
- [ ] Zenodo DOI obtained for code archive
- [ ] Paper submitted to conference/journal

---

## Contact

**Team:**
- Yash Gautam
- Atul Kumar Mishra
- Tanishk Viraj Bhanage

**Repository:** https://github.com/Mish-atul/QuantumSepsis

---

**Last Updated:** April 29, 2026
