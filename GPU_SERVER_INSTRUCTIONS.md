# GPU Server — Final Validation Instructions

> **URGENT:** Server access expires in ~1-2 days  
> **Time Required:** ~50 minutes total  
> **Status:** Ready to execute

---

## Pre-Flight Checklist

Before starting, verify these files exist on the GPU server:

```bash
ssh csegpuserver@172.16.18.2
cd ~/QuantumSepsis

# Check critical files
ls -lh checkpoints/lstm_best.pt                          # LSTM checkpoint
ls -lh data/processed/features.h5                        # Windowed data
ls -lh data/processed/normalization_stats.json           # For Red Team
ls -lh data/processed/test_features.parquet              # For stay-level metrics
```

All should exist. If any are missing, see troubleshooting section below.

---

## Option 1: Automated Execution (RECOMMENDED)

Run all 3 tasks with a single command:

```bash
cd ~/QuantumSepsis
git pull origin main
bash scripts/run_final_validation.sh
```

This will:
1. Run E2E validation (30 min)
2. Compute stay-level metrics (5 min)
3. Generate FINAL_RESULTS.md (1 min)

Monitor progress in real-time. The script will print status updates.

---

## Option 2: Manual Step-by-Step Execution

If you prefer to run each task separately:

### Step 1: E2E Validation (30 min)

```bash
cd ~/QuantumSepsis
export PYTHONPATH=.
python3 scripts/run_e2e_validation.py
```

**Expected output:**
```
Step 1/5 — Loading infrastructure
Step 2/5 — LSTM inference on test set
Step 3/5 — RedTeamAgent evaluation
Step 4/5 — Orchestrator fusion
Step 5/5 — Computing metrics

--- E2E Validation Metrics ---
Alert distribution:  WATCH=X  AMBER=Y  CRITICAL=Z  FT=W
Sensitivity (CRITICAL vs sepsis): 0.XXXX
...
```

**Verify:** Alert distribution should be realistic (not 100% CRITICAL)

### Step 2: Stay-Level Metrics (5 min)

```bash
python3 scripts/compute_stay_level_metrics.py
```

**Expected output:**
```
STAY-LEVEL METRICS
Number of stays:        XXXXX
Sepsis stays:           XXXX (XX.X%)
Stay-level AUROC:       0.XXXX
Stay-level AUPRC:       0.XXXX
...
```

**Verify:** AUROC should be ~0.85-0.88

### Step 3: Final Results Report (1 min)

```bash
python3 scripts/generate_final_results.py
```

**Expected output:**
```
GENERATING FINAL RESULTS REPORT
Loading results from all pipeline stages...
✓ Report saved → FINAL_RESULTS.md

KEY METRICS SUMMARY
Stay-level AUROC:       0.XXXX
Stay-level AUPRC:       0.XXXX
...
```

---

## Verification

After execution, verify these files were created:

```bash
ls -lh data/processed/e2e_validation_results.json
ls -lh data/processed/e2e_decisions.npz
ls -lh data/processed/stay_level_metrics.json
ls -lh FINAL_RESULTS.md
```

Check the key metrics:

```bash
# E2E validation
cat data/processed/e2e_validation_results.json | grep -A 5 "alert_distribution"

# Stay-level metrics
cat data/processed/stay_level_metrics.json | grep "stay_level_auroc"
```

---

## Expected Results

### Alert Distribution (E2E)

| Alert Level | Expected % |
|-------------|-----------|
| WATCH | 50-60% |
| AMBER | 30-40% |
| CRITICAL | 5-15% |

**Red Flag:** If CRITICAL > 50%, something is wrong with denormalization.

### Stay-Level Metrics

| Metric | Expected Range |
|--------|---------------|
| AUROC | 0.85 - 0.88 |
| AUPRC | 0.40 - 0.60 |
| Sensitivity | 0.75 - 0.85 |
| Specificity | 0.85 - 0.92 |
| NPV | 0.95 - 0.98 |

---

## Commit & Push

After successful execution:

```bash
cd ~/QuantumSepsis

# Check what changed
git status

# Add all results
git add data/processed/e2e_validation_results.json
git add data/processed/e2e_decisions.npz
git add data/processed/stay_level_metrics.json
git add FINAL_RESULTS.md

# Commit
git commit -m "Final results: E2E validation, stay-level metrics, complete report

- E2E validation with fixed Red Team denormalization
- Alert distribution: WATCH XX%, AMBER XX%, CRITICAL XX%
- Stay-level AUROC: 0.XXXX, AUPRC: 0.XXXX
- Sensitivity: 0.XXXX, Specificity: 0.XXXX
- FINAL_RESULTS.md generated with all metrics"

# Push to GitHub
git push origin main
```

---

## Troubleshooting

### Issue: "File not found: conformal_calibration.json"

**Cause:** Conformal calibration hasn't been run yet.

**Solution:**
```bash
python3 scripts/run_conformal_calibration.py
```

This should take ~5 minutes.

### Issue: "File not found: lstm_best.pt"

**Cause:** LSTM training incomplete or checkpoint missing.

**Solution:** Check if training completed:
```bash
ls -lh checkpoints/
cat logs/lstm_training_*.log | tail -50
```

If checkpoint exists but in different location, update path in script.

### Issue: "Red Team shows 100% CRITICAL"

**Cause:** Normalization stats missing or incorrect format.

**Solution:** Verify normalization stats:
```bash
cat data/processed/normalization_stats.json | head -30
```

Should show:
```json
{
  "train_mean": {
    "heart_rate": 85.XX,
    "sbp": 120.XX,
    ...
  },
  "train_std": {
    "heart_rate": 15.XX,
    ...
  }
}
```

If missing, re-run preprocessing:
```bash
python3 -m src.data.preprocessing \
    --train data/processed/train_features.parquet \
    --val data/processed/val_features.parquet \
    --test data/processed/test_features.parquet
```

### Issue: "Stay-level script fails with KeyError"

**Cause:** test_features.parquet missing or wrong format.

**Solution:** Check if file exists:
```bash
ls -lh data/processed/test_features.parquet
python3 -c "import pandas as pd; df = pd.read_parquet('data/processed/test_features.parquet'); print(df.columns.tolist())"
```

Should include 'stay_id' column.

### Issue: "Out of memory"

**Cause:** E2E validation loading too much data at once.

**Solution:** Reduce batch size:
```bash
python3 scripts/run_e2e_validation.py --batch-size 256
```

---

## After GPU Server Work

Once these tasks are complete, **all GPU server work is DONE**. The server can expire without impact.

### Remaining Work (Local Machine)

1. **Test Streamlit dashboard:**
   ```bash
   # Copy models from server first
   scp csegpuserver@172.16.18.2:~/QuantumSepsis/checkpoints/lstm_best.pt checkpoints/
   scp csegpuserver@172.16.18.2:~/QuantumSepsis/data/processed/conformal_calibration.json data/processed/
   scp csegpuserver@172.16.18.2:~/QuantumSepsis/data/processed/normalization_stats.json data/processed/
   
   # Run dashboard
   pip install streamlit plotly
   streamlit run scripts/realtime_demo.py
   ```

2. **Take screenshots** of all 3 patient scenarios

3. **Review FINAL_RESULTS.md** and update if needed

4. **Write paper** using FINAL_RESULTS.md as source

5. **Create presentation slides**

---

## Timeline

| Task | Duration | Blocking? |
|------|----------|-----------|
| E2E Validation | 30 min | Yes (GPU) |
| Stay-Level Metrics | 5 min | Yes (needs E2E) |
| Final Report | 1 min | Yes (needs all) |
| Commit & Push | 2 min | No |
| **Total** | **~40 min** | — |

---

## Success Criteria

✅ All 4 output files created  
✅ Alert distribution realistic (not 100% CRITICAL)  
✅ Stay-level AUROC ≥ 0.85  
✅ Stay-level AUPRC ≥ 0.40  
✅ Sensitivity ≥ 0.75  
✅ NPV ≥ 0.95  
✅ FINAL_RESULTS.md generated  
✅ Committed and pushed to GitHub  

---

## Contact

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Check `FINAL_TASKS_README.md` for detailed explanations
4. Contact team members

---

**Last Updated:** April 29, 2026  
**Status:** Ready to execute
