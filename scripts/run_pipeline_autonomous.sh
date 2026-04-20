#!/usr/bin/env bash
set -euo pipefail

cd ~/QuantumSepsis
mkdir -p logs data/processed

LOG="logs/pipeline_autonomous.log"
exec > >(tee -a "$LOG") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

VENV_PY="$HOME/QuantumSepsis/.venv/bin/python"

wait_for_venv() {
  log "Waiting for venv dependencies to become available..."
  while true; do
    if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import pandas,numpy,scipy,sklearn,h5py,pyarrow,torch" >/dev/null 2>&1; then
      "$VENV_PY" -c "import torch; print('venv_torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
      log "Venv is ready."
      return
    fi
    log "Venv not ready yet; retrying in 60s."
    sleep 60
  done
}

wait_for_cohort() {
  log "Waiting for data/processed/cohort.csv from cohort extraction..."
  while [[ ! -f data/processed/cohort.csv ]]; do
    if ! pgrep -f "src.data.cohort_extraction_optimized" >/dev/null 2>&1; then
      log "ERROR: Cohort extraction process not running and cohort.csv is missing."
      exit 1
    fi
    tail -n 4 logs/cohort_extraction_optimized.log || true
    sleep 60
  done
  log "Detected data/processed/cohort.csv"
  wc -l data/processed/cohort.csv
  "$VENV_PY" - <<'PY'
import pandas as pd

c = pd.read_csv("data/processed/cohort.csv")
prev = float(c["sepsis_label"].mean()) if "sepsis_label" in c.columns else float("nan")
print(f"cohort_rows={len(c)}")
print(f"cohort_sepsis_prevalence={prev:.6f}")
PY
}

run_feature_extraction() {
  log "Starting feature extraction..."
  "$VENV_PY" -m src.data.feature_extraction --cohort data/processed/cohort.csv --data-dir data/raw/physionet.org/files/mimiciv/3.1
  "$VENV_PY" - <<'PY'
import pandas as pd

f = pd.read_parquet("data/processed/hourly_features.parquet")
print(f"hourly_rows={len(f)}")
print(f"hourly_stays={f['stay_id'].nunique()}")
PY
}

run_preprocessing() {
  log "Starting preprocessing..."
  "$VENV_PY" -m src.data.preprocessing --features data/processed/hourly_features.parquet --cohort data/processed/cohort.csv
  "$VENV_PY" - <<'PY'
import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/processed/{split}_features.parquet")
    print(f"{split}_rows={len(df)} {split}_stays={df['stay_id'].nunique()}")
PY
}

create_windowing_runner() {
  log "Creating run_windowing.py for real-data window generation..."
  cat > run_windowing.py <<'PY'
import logging
import pandas as pd

from src.config import get_default_config
from src.data.windowing import WindowGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

config = get_default_config()
generator = WindowGenerator(config)

cohort = pd.read_csv(
    "data/processed/cohort.csv",
    parse_dates=["intime", "outtime", "sepsis_onset_time"],
)

results = {}
for split in ["train", "val", "test"]:
    features = pd.read_parquet(f"data/processed/{split}_features.parquet")
    split_stays = set(features["stay_id"].unique())
    split_cohort = cohort[cohort["stay_id"].isin(split_stays)]

    X, y, meta = generator.generate_windows(features, split_cohort, split)
    results[split] = (X, y, meta)

    print(f"{split}: X={X.shape}, positives={int(y.sum())}, positive_rate={float(y.mean()):.6f}")

generator.save_to_hdf5(
    results["train"][0],
    results["train"][1],
    results["val"][0],
    results["val"][1],
    results["test"][0],
    results["test"][1],
    results["train"][2],
    results["val"][2],
    results["test"][2],
)
PY
}

run_windowing() {
  log "Running window generation..."
  "$VENV_PY" run_windowing.py
  "$VENV_PY" - <<'PY'
import h5py

with h5py.File("data/processed/features.h5", "r") as f:
    for split in ["train", "val", "test"]:
        X = f[f"X_{split}"]
        y = f[f"y_{split}"][:]
        print(
            f"{split}_windows={X.shape[0]} {split}_shape={tuple(X.shape)} "
            f"positives={int(y.sum())} positive_rate={float(y.mean()):.6f}"
        )
PY
}

run_training() {
  log "Starting LSTM training on CUDA_VISIBLE_DEVICES=0..."
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled "$VENV_PY" -m src.training.train_lstm --data data/processed/features.h5
}

collect_final_metrics() {
  log "Collecting final metrics and stage summaries..."
  CUDA_VISIBLE_DEVICES=0 "$VENV_PY" - <<'PY'
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from src.config import get_default_config
from src.evaluation.metrics import compute_all_metrics
from src.models.lstm import SepsisLSTM

summary = {}

cohort = pd.read_csv("data/processed/cohort.csv")
summary["cohort_rows"] = int(len(cohort))
summary["cohort_sepsis_prevalence"] = float(cohort["sepsis_label"].mean())

hourly = pd.read_parquet("data/processed/hourly_features.parquet")
summary["hourly_rows"] = int(len(hourly))
summary["hourly_stays"] = int(hourly["stay_id"].nunique())

for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/processed/{split}_features.parquet")
    summary[f"{split}_rows"] = int(len(df))
    summary[f"{split}_stays"] = int(df["stay_id"].nunique())

with h5py.File("data/processed/features.h5", "r") as f:
    for split in ["train", "val", "test"]:
        X = f[f"X_{split}"][:]
        y = f[f"y_{split}"][:]
        summary[f"{split}_windows"] = int(X.shape[0])
        summary[f"{split}_positive_windows"] = int(y.sum())
        summary[f"{split}_positive_rate"] = float(y.mean())

with h5py.File("data/processed/features.h5", "r") as f:
    X_test = f["X_test"][:].astype(np.float32)
    y_test = f["y_test"][:].astype(np.int8)

config = get_default_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SepsisLSTM(config.lstm).to(device)
checkpoint = torch.load("checkpoints/lstm_best.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scores = []
batch_size = 2048
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[i : i + batch_size]).to(device)
        out = model(xb)
        scores.append(out["risk_score"].squeeze(-1).cpu().numpy())

y_score = np.concatenate(scores)
metrics = compute_all_metrics(y_test, y_score, prefix="test_")

wanted = {
    "test_auroc": metrics.get("test_auroc"),
    "test_auprc": metrics.get("test_auprc"),
    "test_sensitivity_at_95spec": metrics.get("test_sensitivity_at_95spec"),
    "test_specificity_at_95sens": metrics.get("test_specificity_at_95sens"),
    "test_f1": metrics.get("test_f1"),
}

summary["test_metrics"] = wanted

out_path = Path("data/processed/pipeline_results_real.json")
out_path.write_text(json.dumps(summary, indent=2))

print("FINAL_TEST_METRICS", json.dumps(wanted, indent=2))
print("SUMMARY_WRITTEN", str(out_path))
PY
}

log "Autonomous pipeline runner started."
wait_for_venv
wait_for_cohort
run_feature_extraction
run_preprocessing
create_windowing_runner
run_windowing
run_training
collect_final_metrics
log "Autonomous pipeline completed successfully."