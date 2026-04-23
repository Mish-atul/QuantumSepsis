"""Run real-data baselines and summarize final comparison metrics."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from src.baselines.sofa_baseline import SOFABaseline
from src.baselines.xgboost_baseline import XGBoostBaseline
from src.config import get_default_config
from src.evaluation.metrics import compute_all_metrics
from src.models.lstm import SepsisLSTM


def evaluate_lstm(config, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path("checkpoints/lstm_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SepsisLSTM(config.lstm).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_size = 2048
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i : i + batch_size].astype(np.float32)).to(device)
            out = model(xb)
            scores.append(out["risk_score"].squeeze(-1).cpu().numpy())

    y_score = np.concatenate(scores)
    metrics = compute_all_metrics(y_test.astype(np.int8), y_score, prefix="lstm_test_")

    val_metrics = checkpoint.get("metrics", {})
    return {
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "val_auroc": float(val_metrics.get("val_auroc", np.nan)),
        "val_auprc": float(val_metrics.get("val_auprc", np.nan)),
        "test_auroc": float(metrics.get("lstm_test_auroc", np.nan)),
        "test_auprc": float(metrics.get("lstm_test_auprc", np.nan)),
        "test_sensitivity_at_95spec": float(metrics.get("lstm_test_sensitivity_at_95spec", np.nan)),
    }


def evaluate_xgboost(config, X_train, y_train, X_val, y_val, X_test, y_test):
    baseline = XGBoostBaseline(config)
    baseline.train(X_train, y_train, X_val, y_val)
    metrics = baseline.evaluate(X_test, y_test)
    return {
        "test_auroc": float(metrics.get("xgb_test_auroc", np.nan)),
        "test_auprc": float(metrics.get("xgb_test_auprc", np.nan)),
    }


def evaluate_sofa(X_test, y_test):
    baseline = SOFABaseline(threshold=2.0)
    metrics = baseline.evaluate(X_test, y_test)
    return {
        "test_auroc": float(metrics.get("sofa_test_auroc", np.nan)),
        "test_auprc": float(metrics.get("sofa_test_auprc", np.nan)),
    }


def markdown_table(results):
    rows = [
        "| Model | Test AUROC | Test AUPRC | Sensitivity@95%Spec |",
        "|---|---:|---:|---:|",
        (
            "| Classical LSTM (Real) | "
            f"{results['lstm']['test_auroc']:.4f} | "
            f"{results['lstm']['test_auprc']:.4f} | "
            f"{results['lstm']['test_sensitivity_at_95spec']:.4f} |"
        ),
        (
            "| XGBoost Baseline (Real) | "
            f"{results['xgboost']['test_auroc']:.4f} | "
            f"{results['xgboost']['test_auprc']:.4f} | "
            "- |"
        ),
        (
            "| SOFA Baseline (Real) | "
            f"{results['sofa']['test_auroc']:.4f} | "
            f"{results['sofa']['test_auprc']:.4f} | "
            "- |"
        ),
    ]
    return "\n".join(rows)


def main():
    config = get_default_config()

    cohort = pd.read_csv("data/processed/cohort.csv")
    hourly = pd.read_parquet("data/processed/hourly_features.parquet")

    with h5py.File("data/processed/features.h5", "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_val = f["X_val"][:]
        y_val = f["y_val"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]

    results = {
        "cohort": {
            "n_stays": int(len(cohort)),
            "sepsis_count": int(cohort["sepsis_label"].sum()),
            "sepsis_prevalence": float(cohort["sepsis_label"].mean()),
        },
        "feature_matrix": {
            "shape": [int(len(hourly)), int(hourly.shape[1])],
            "n_stays": int(hourly["stay_id"].nunique()),
        },
        "windows": {
            "train_shape": [int(v) for v in X_train.shape],
            "val_shape": [int(v) for v in X_val.shape],
            "test_shape": [int(v) for v in X_test.shape],
        },
    }

    results["lstm"] = evaluate_lstm(config, X_test, y_test)
    results["xgboost"] = evaluate_xgboost(config, X_train, y_train, X_val, y_val, X_test, y_test)
    results["sofa"] = evaluate_sofa(X_test, y_test)

    out_json = Path("data/processed/pipeline_results_real.json")
    out_md = Path("data/processed/pipeline_results_real.md")

    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text(markdown_table(results) + "\n")

    print(json.dumps(results, indent=2))
    print("\n" + markdown_table(results))


if __name__ == "__main__":
    main()