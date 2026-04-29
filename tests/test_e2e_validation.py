"""
Edge case tests for run_e2e_validation.py

Run with:
    python3 tests/test_e2e_validation.py
    python3 -m pytest tests/test_e2e_validation.py -v
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TESTS = []

def test(fn):
    TESTS.append(fn)
    return fn

def run_all():
    passed, failed = 0, []
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {name}")
            print(f"       {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {len(failed)} failed")
    if failed:
        print(f"Failed:  {failed}")
    return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_h5(path, X_train, y_train, X_val, y_val, X_test, y_test):
    import h5py
    with h5py.File(path, "w") as f:
        for split, X, y in [("train",X_train,y_train),
                             ("val",  X_val,  y_val),
                             ("test", X_test, y_test)]:
            f.create_dataset(f"X_{split}", data=X)
            f.create_dataset(f"y_{split}", data=y)


def _write_checkpoint(path, model):
    import torch
    torch.save({"epoch":0,"model_state_dict":model.state_dict(),
                "best_val_auroc":0.5,"metrics":{}}, path)


def _write_cal_json(path, q_alpha=0.25):
    with open(path,"w") as f:
        json.dump({"q_alpha":q_alpha,"coverage_guarantee":0.90,
                   "method":"split_conformal"}, f)


def _make_synthetic(n_test=100, pos_rate=0.13, seed=42):
    """Returns (X_train,y_train,X_val,y_val,X_test,y_test) as np arrays."""
    rng = np.random.default_rng(seed)
    def split(n):
        y = (rng.random(n) < pos_rate).astype(np.float32)
        X = rng.standard_normal((n,6,12)).astype(np.float32)
        X[y==1] += 0.3
        return X, y.astype(np.int8)
    return (*split(200), *split(80), *split(n_test))


def _run_pipeline(tmpdir, q_alpha=0.25, n_test=100, pos_rate=0.13,
                  batch_size=16, device_str="cpu", seed=42,
                  write_norm_stats=False):
    """Helper: set up temp files and run the full pipeline."""
    import torch
    from src.models.lstm import SepsisLSTM
    from src.config import get_default_config
    from scripts.run_e2e_validation import run_e2e_validation

    config  = get_default_config()
    model   = SepsisLSTM(config.lstm)
    tmpdir  = Path(tmpdir)

    data = _make_synthetic(n_test=n_test, pos_rate=pos_rate, seed=seed)
    h5   = tmpdir / "features.h5"
    _write_h5(h5, *data)

    ckpt = tmpdir / "lstm_best.pt"
    _write_checkpoint(ckpt, model)

    cal  = tmpdir / "conformal_calibration.json"
    _write_cal_json(cal, q_alpha=q_alpha)

    norm = tmpdir / "norm_stats.json"
    if write_norm_stats:
        # Minimal norm stats structure matching preprocessing.py output
        feature_names = ["heart_rate","sbp","dbp","map","temperature",
                         "resp_rate","spo2","gcs_total",
                         "lactate","wbc","creatinine","platelets"]
        stats = {
            "train_mean": {f: 0.0 for f in feature_names},
            "train_std" : {f: 1.0 for f in feature_names},
        }
        with open(norm,"w") as fh:
            json.dump(stats, fh)

    out = tmpdir / "out"

    return run_e2e_validation(
        checkpoint_path       = str(ckpt),
        data_path             = str(h5),
        calibration_json_path = str(cal),
        norm_stats_path       = str(norm) if write_norm_stats else str(tmpdir/"missing.json"),
        output_dir            = str(out),
        batch_size            = batch_size,
        device_str            = device_str,
    ), out


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_full_pipeline_runs_and_outputs_files():
    """Normal run produces all required output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, out = _run_pipeline(tmpdir)
        assert (out / "e2e_validation_results.json").exists(), "JSON missing"
        assert (out / "e2e_decisions.npz").exists(), "NPZ missing"


@test
def test_result_has_required_keys():
    """Result dict must contain all required top-level keys."""
    required = ["n_total","alert_distribution","clinical_metrics",
                "continuous_metrics","confidence_stats","red_team_stats"]
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir)
        for k in required:
            assert k in result, f"Missing key: {k}"


@test
def test_alert_distribution_sums_to_n():
    """WATCH + AMBER + CRITICAL counts must equal total windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=200)
        dist = result["alert_distribution"]
        total = dist["WATCH"] + dist["AMBER"] + dist["CRITICAL"]
        assert total == result["n_total"], \
            f"Alert counts {total} != n_total {result['n_total']}"


@test
def test_sensitivity_in_range():
    """Sensitivity must be in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=150, pos_rate=0.20)
        s = result["clinical_metrics"]["sensitivity_at_critical"]
        assert 0.0 <= s <= 1.0, f"Sensitivity {s} out of [0,1]"


@test
def test_npz_arrays_correct_length():
    """NPZ arrays must have exactly n_test entries."""
    n = 120
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out = _run_pipeline(tmpdir, n_test=n)
        data = np.load(out / "e2e_decisions.npz")
        for key in ["risk_scores","alert_labels","true_labels",
                    "conformal_widths","confidences","fast_tracked"]:
            assert len(data[key]) == n, \
                f"Array '{key}' has length {len(data[key])}, expected {n}"


@test
def test_alert_labels_valid_values():
    """alert_labels must only contain 0, 1, or 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, out = _run_pipeline(tmpdir, n_test=100)
        labels = np.load(out / "e2e_decisions.npz")["alert_labels"]
        assert set(labels.tolist()).issubset({0,1,2}), \
            f"Unexpected alert_label values: {set(labels.tolist())}"


@test
def test_missing_calibration_json_raises():
    """Missing conformal_calibration.json must raise FileNotFoundError."""
    import torch
    from src.models.lstm import SepsisLSTM
    from src.config import get_default_config
    from scripts.run_e2e_validation import run_e2e_validation

    config = get_default_config()
    model  = SepsisLSTM(config.lstm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = _make_synthetic(n_test=50)
        h5 = tmpdir/"features.h5"
        _write_h5(h5, *data)
        ckpt = tmpdir/"lstm_best.pt"
        _write_checkpoint(ckpt, model)
        raised = False
        try:
            run_e2e_validation(
                checkpoint_path       = str(ckpt),
                data_path             = str(h5),
                calibration_json_path = str(tmpdir/"missing.json"),
                norm_stats_path       = str(tmpdir/"missing2.json"),
                output_dir            = str(tmpdir/"out"),
                device_str            = "cpu",
            )
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing calibration JSON"


@test
def test_missing_checkpoint_raises():
    """Missing LSTM checkpoint must raise FileNotFoundError."""
    from scripts.run_e2e_validation import run_e2e_validation
    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = _make_synthetic(n_test=50)
        h5 = tmpdir/"features.h5"
        _write_h5(h5, *data)
        cal = tmpdir/"cal.json"
        _write_cal_json(cal)
        raised = False
        try:
            run_e2e_validation(
                checkpoint_path       = str(tmpdir/"missing_checkpoint.pt"),
                data_path             = str(h5),
                calibration_json_path = str(cal),
                norm_stats_path       = str(tmpdir/"missing.json"),
                output_dir            = str(tmpdir/"out"),
                device_str            = "cpu",
            )
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing checkpoint"


@test
def test_missing_hdf5_raises():
    """Missing features.h5 must raise FileNotFoundError."""
    import torch
    from src.models.lstm import SepsisLSTM
    from src.config import get_default_config
    from scripts.run_e2e_validation import run_e2e_validation

    config = get_default_config()
    model  = SepsisLSTM(config.lstm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt = tmpdir/"lstm_best.pt"
        _write_checkpoint(ckpt, model)
        cal = tmpdir/"cal.json"
        _write_cal_json(cal)
        raised = False
        try:
            run_e2e_validation(
                checkpoint_path       = str(ckpt),
                data_path             = str(tmpdir/"missing.h5"),
                calibration_json_path = str(cal),
                norm_stats_path       = str(tmpdir/"missing.json"),
                output_dir            = str(tmpdir/"out"),
                device_str            = "cpu",
            )
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing HDF5"


@test
def test_all_negative_labels_no_crash():
    """Test set with zero sepsis cases must not crash (sensitivity = 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100, pos_rate=0.0)
        assert result["n_sepsis_positive"] == 0
        assert result["clinical_metrics"]["sensitivity_at_critical"] == 0.0


@test
def test_all_positive_labels_no_crash():
    """Test set with 100% sepsis must not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100, pos_rate=1.0)
        assert result["n_sepsis_positive"] == 100


@test
def test_very_wide_q_alpha_forces_amber_escalations():
    """q_alpha=0.5 → all intervals width=1.0 → many WATCH→AMBER escalations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100, q_alpha=0.5)
        # Width = min(1, score+0.5) - max(0, score-0.5) ≈ 1.0 for mid scores
        # → confidence = 0 → AMBER escalation for low-risk windows
        n_watch = result["alert_distribution"]["WATCH"]
        n_amber = result["alert_distribution"]["AMBER"]
        # With q_alpha=0.5, many WATCH cases get escalated to AMBER
        assert n_amber > 0, "Expected some AMBER escalations with wide q_alpha"


@test
def test_zero_q_alpha_no_escalations_from_width():
    """q_alpha=0.0 → zero-width intervals → no uncertainty escalations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, out = _run_pipeline(tmpdir, n_test=100, q_alpha=0.0)
        # Zero-width intervals → confidence=1.0 → no WATCH→AMBER from uncertainty
        data = np.load(out / "e2e_decisions.npz")
        widths = data["conformal_widths"]
        assert (widths == 0.0).all(), f"Expected all zero widths, got max={widths.max()}"


@test
def test_with_real_norm_stats():
    """Pipeline with valid normalization_stats.json runs correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100, write_norm_stats=True)
        assert result["n_total"] == 100


@test
def test_batch_size_one_vs_large_same_scores():
    """Batch size 1 and 512 must produce identical risk scores."""
    import torch
    from src.models.lstm import SepsisLSTM
    from src.config import get_default_config
    from scripts.run_e2e_validation import run_e2e_validation

    config = get_default_config()
    model  = SepsisLSTM(config.lstm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = _make_synthetic(n_test=40, seed=77)
        h5   = tmpdir/"features.h5";  _write_h5(h5, *data)
        ckpt = tmpdir/"lstm_best.pt"; _write_checkpoint(ckpt, model)
        cal  = tmpdir/"cal.json";     _write_cal_json(cal)

        def run(bs, outname):
            return run_e2e_validation(
                checkpoint_path       = str(ckpt),
                data_path             = str(h5),
                calibration_json_path = str(cal),
                norm_stats_path       = str(tmpdir/"missing.json"),
                output_dir            = str(tmpdir/outname),
                batch_size            = bs, device_str="cpu",
            )

        r1 = run(1,   "out1");  d1 = np.load(tmpdir/"out1"/"e2e_decisions.npz")
        r2 = run(512, "out2");  d2 = np.load(tmpdir/"out2"/"e2e_decisions.npz")

        np.testing.assert_allclose(
            d1["risk_scores"], d2["risk_scores"], atol=1e-5,
            err_msg="risk_scores differ between batch_size=1 and batch_size=512"
        )


@test
def test_pct_sepsis_missed_calculation():
    """pct_sepsis_missed = fn_at_watch / n_sepsis_positive * 100."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=200, pos_rate=0.20)
        cm  = result["clinical_metrics"]
        n_pos = result["n_sepsis_positive"]
        if n_pos > 0:
            expected_pct = round(100 * cm["fn_at_watch"] / n_pos, 2)
            assert abs(cm["pct_sepsis_missed"] - expected_pct) < 0.1, \
                f"pct_sepsis_missed mismatch: got {cm['pct_sepsis_missed']}, expected {expected_pct}"


@test
def test_red_team_override_count_nonnegative():
    """Red Team override count must be >= 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100)
        assert result["red_team_stats"]["n_overrides"] >= 0


@test
def test_confidence_stats_in_range():
    """Mean confidence and pct_low/high_confidence must be in valid ranges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _run_pipeline(tmpdir, n_test=100)
        cs = result["confidence_stats"]
        assert 0.0 <= cs["mean_confidence"]      <= 1.0
        assert 0.0 <= cs["pct_low_confidence"]  <= 100.0
        assert 0.0 <= cs["pct_high_confidence"] <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("QuantumSepsis — E2E Validation Edge Case Tests")
    print("="*60)
    ok = run_all()
    sys.exit(0 if ok else 1)
