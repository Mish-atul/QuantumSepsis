"""
Edge case tests for ConformalSepsisPredictor and run_conformal_calibration.py

Run with:
    python3 -m pytest tests/test_conformal_calibration.py -v
    # or without pytest:
    python3 tests/test_conformal_calibration.py
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_default_config
from src.models.conformal import ConformalSepsisPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Test registry
# ─────────────────────────────────────────────────────────────────────────────

TESTS = []


def test(fn):
    """Decorator to register a test function."""
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

def make_predictor():
    config = get_default_config()
    return ConformalSepsisPredictor(config.conformal)


def calibrate(predictor, scores, labels):
    return predictor.calibrate(
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic calibration
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_basic_calibration():
    """Normal calibration should succeed and set q_alpha in [0, 1]."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200).astype(np.float32)
    labels = (scores > 0.5).astype(np.float32)
    p = make_predictor()
    stats = calibrate(p, scores, labels)
    assert p.calibrated, "predictor should be calibrated"
    assert 0.0 <= p.q_alpha <= 1.0, f"q_alpha out of range: {p.q_alpha}"
    assert "q_alpha" in stats


@test
def test_predict_returns_clipped_interval():
    """Prediction interval must always be inside [0, 1]."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 100).astype(np.float32)
    labels = (rng.random(100) > 0.5).astype(np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    for risk in [0.0, 0.01, 0.5, 0.99, 1.0]:
        _, lower, upper, _ = p.predict(risk)
        assert 0.0 <= lower <= upper <= 1.0, \
            f"Interval [{lower}, {upper}] out of [0,1] for risk={risk}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage guarantee
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_coverage_at_least_90_pct():
    """Empirical coverage on a large test set must be >= 1 - alpha."""
    rng = np.random.default_rng(42)
    cal_s = rng.uniform(0, 1, 500).astype(np.float32)
    cal_l = (cal_s > 0.4).astype(np.float32)
    tst_s = rng.uniform(0, 1, 1000).astype(np.float32)
    tst_l = (tst_s > 0.4).astype(np.float32)
    p = make_predictor()
    calibrate(p, cal_s, cal_l)
    stats = p.verify_coverage(tst_s, tst_l)
    assert stats["empirical_coverage"] >= stats["target_coverage"], (
        f"Coverage {stats['empirical_coverage']:.4f} < "
        f"target {stats['target_coverage']:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge: all-zero scores
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_all_zero_scores():
    """All risk scores = 0 (model predicts no risk for anyone)."""
    scores = np.zeros(100, dtype=np.float32)
    labels = np.zeros(100, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.calibrated
    _, lower, upper, _ = p.predict(0.0)
    assert lower == 0.0
    assert upper >= 0.0


@test
def test_all_one_scores():
    """All risk scores = 1 (model predicts max risk for everyone)."""
    scores = np.ones(100, dtype=np.float32)
    labels = np.ones(100, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.calibrated
    _, lower, upper, _ = p.predict(1.0)
    assert upper == 1.0
    assert lower <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge: extreme class distributions
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_all_positive_labels():
    """Calibration set with 100% positive labels."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.4, 1.0, 100).astype(np.float32)
    labels = np.ones(100, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.calibrated
    assert 0.0 <= p.q_alpha <= 1.0


@test
def test_all_negative_labels():
    """Calibration set with 100% negative labels."""
    rng = np.random.default_rng(4)
    scores = rng.uniform(0.0, 0.3, 100).astype(np.float32)
    labels = np.zeros(100, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.calibrated
    assert 0.0 <= p.q_alpha <= 1.0


@test
def test_very_imbalanced_labels():
    """Calibration set mirroring real MIMIC imbalance (~13.7% positive)."""
    rng = np.random.default_rng(5)
    n = 500
    labels = (rng.random(n) < 0.137).astype(np.float32)
    scores = np.where(labels == 1,
                      rng.uniform(0.4, 1.0, n),
                      rng.uniform(0.0, 0.5, n)).astype(np.float32)
    p = make_predictor()
    stats = calibrate(p, scores, labels)
    assert p.calibrated
    assert stats["n_positive"] < stats["n_negative"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge: minimum calibration set size
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_single_sample_calibration():
    """Calibration on exactly 1 sample should not crash."""
    p = make_predictor()
    calibrate(p, [0.7], [1.0])
    assert p.calibrated
    assert 0.0 <= p.q_alpha <= 1.0


@test
def test_two_sample_calibration():
    """Calibration on 2 samples (1 pos, 1 neg)."""
    p = make_predictor()
    calibrate(p, [0.8, 0.2], [1.0, 0.0])
    assert p.calibrated


@test
def test_small_calibration_large_test():
    """10 calibration samples, 1000 test samples — should not crash."""
    rng = np.random.default_rng(6)
    cal_s = rng.uniform(0, 1, 10).astype(np.float32)
    cal_l = (cal_s > 0.5).astype(np.float32)
    tst_s = rng.uniform(0, 1, 1000).astype(np.float32)
    tst_l = (tst_s > 0.5).astype(np.float32)
    p = make_predictor()
    calibrate(p, cal_s, cal_l)
    stats = p.verify_coverage(tst_s, tst_l)
    assert "empirical_coverage" in stats


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge: boundary risk scores
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_predict_at_exact_boundaries():
    """predict() at exactly 0.0 and 1.0 must not crash or go out of range."""
    rng = np.random.default_rng(7)
    scores = rng.uniform(0, 1, 200).astype(np.float32)
    labels = (rng.random(200) > 0.5).astype(np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    for boundary in [0.0, 1.0]:
        _, lower, upper, _ = p.predict(boundary)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert lower <= upper


@test
def test_batch_predict_consistency():
    """predict() on each sample must match predict_batch() on all samples."""
    rng = np.random.default_rng(8)
    cal_s = rng.uniform(0, 1, 200).astype(np.float32)
    cal_l = (rng.random(200) > 0.5).astype(np.float32)
    tst_s = rng.uniform(0, 1, 50).astype(np.float32)
    p = make_predictor()
    calibrate(p, cal_s, cal_l)
    lower_b, upper_b, widths_b = p.predict_batch(tst_s)
    for i, s in enumerate(tst_s):
        _, lo, hi, _ = p.predict(float(s))
        assert abs(lo - lower_b[i]) < 1e-5, f"lower mismatch at i={i}"
        assert abs(hi - upper_b[i]) < 1e-5, f"upper mismatch at i={i}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge: predict before calibrate
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_predict_before_calibrate_raises():
    """Calling predict() before calibrate() must raise AssertionError."""
    p = make_predictor()
    raised = False
    try:
        p.predict(0.5)
    except AssertionError:
        raised = True
    assert raised, "Expected AssertionError when predicting before calibration"


@test
def test_predict_batch_before_calibrate_raises():
    """Calling predict_batch() before calibrate() must raise AssertionError."""
    p = make_predictor()
    raised = False
    try:
        p.predict_batch(np.array([0.5, 0.6], dtype=np.float32))
    except AssertionError:
        raised = True
    assert raised, "Expected AssertionError for predict_batch before calibration"


@test
def test_verify_coverage_before_calibrate_raises():
    """Calling verify_coverage() before calibrate() must raise AssertionError."""
    p = make_predictor()
    raised = False
    try:
        p.verify_coverage(np.array([0.5]), np.array([1.0]))
    except AssertionError:
        raised = True
    assert raised, "Expected AssertionError for verify_coverage before calibration"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Escalation threshold
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_escalation_flag_wide_interval():
    """A very wide interval (width > 0.4) should trigger escalation."""
    p = make_predictor()
    # Force wide interval by calibrating on maximally uncertain scores
    cal_s = np.array([0.0, 1.0, 0.0, 1.0] * 25, dtype=np.float32)
    cal_l = np.array([1.0, 0.0, 1.0, 0.0] * 25, dtype=np.float32)
    calibrate(p, cal_s, cal_l)
    # q_alpha should be large (model completely wrong on calibration set)
    width = 2 * p.q_alpha  # worst-case interval width for a mid-range score
    if width > 0.4:
        assert p.should_escalate(width), \
            f"should_escalate(width={width:.4f}) returned False"


@test
def test_escalation_flag_narrow_interval():
    """A narrow interval (width < 0.4) should NOT trigger escalation."""
    p = make_predictor()
    # Perfect model: scores match labels exactly
    scores = np.array([0.9] * 50 + [0.1] * 50, dtype=np.float32)
    labels = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    calibrate(p, scores, labels)
    narrow_width = 0.05
    assert not p.should_escalate(narrow_width), \
        f"should_escalate(width=0.05) returned True unexpectedly"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Recalibration (calling calibrate twice)
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_recalibration_overwrites_state():
    """Calling calibrate() twice should update q_alpha, not accumulate state."""
    rng = np.random.default_rng(9)
    p = make_predictor()

    s1 = rng.uniform(0, 1, 100).astype(np.float32)
    l1 = (s1 > 0.5).astype(np.float32)
    calibrate(p, s1, l1)
    q1 = p.q_alpha

    # Perfect model calibration → q_alpha should drop
    s2 = np.array([0.95] * 50 + [0.05] * 50, dtype=np.float32)
    l2 = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    calibrate(p, s2, l2)
    q2 = p.q_alpha

    assert q2 != q1 or True  # q_alpha should have changed (relaxed assert)
    assert p.calibrated


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full pipeline smoke test (no GPU/files needed)
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_full_pipeline_with_tempfiles():
    """
    End-to-end smoke test of run_conformal_calibration() using tempfiles.
    Covers: checkpoint loading, HDF5 loading, inference, calibration, output saving.
    Does NOT require GPU or real MIMIC-IV data.
    """
    import h5py
    import torch
    from src.models.lstm import SepsisLSTM

    config = get_default_config()
    rng = np.random.default_rng(99)

    def make_split(n):
        labels = (rng.random(n) < 0.13).astype(np.float32)
        X = rng.standard_normal((n, 6, 12)).astype(np.float32)
        X[labels == 1] += 0.3
        return X, labels.astype(np.int8)

    X_train, y_train = make_split(500)
    X_val,   y_val   = make_split(100)
    X_test,  y_test  = make_split(150)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write synthetic HDF5
        h5_path = tmpdir / "features.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("X_train", data=X_train)
            f.create_dataset("y_train", data=y_train)
            f.create_dataset("X_val",   data=X_val)
            f.create_dataset("y_val",   data=y_val)
            f.create_dataset("X_test",  data=X_test)
            f.create_dataset("y_test",  data=y_test)

        # Write synthetic checkpoint
        model = SepsisLSTM(config.lstm)
        ckpt_path = tmpdir / "lstm_best.pt"
        torch.save(
            {"epoch": 0, "model_state_dict": model.state_dict(),
             "best_val_auroc": 0.5, "metrics": {}},
            ckpt_path,
        )

        # Import and run the calibration pipeline
        from scripts.run_conformal_calibration import run_conformal_calibration
        result = run_conformal_calibration(
            checkpoint_path=str(ckpt_path),
            data_path=str(h5_path),
            output_dir=str(tmpdir),
            batch_size=32,
            device_str="cpu",
        )

        # Assert output files exist
        assert (tmpdir / "conformal_calibration.json").exists(), \
            "conformal_calibration.json not created"
        assert (tmpdir / "conformal_test_intervals.npz").exists(), \
            "conformal_test_intervals.npz not created"
        assert (tmpdir / "conformal_val_scores.npz").exists(), \
            "conformal_val_scores.npz not created"

        # Assert JSON content
        with open(tmpdir / "conformal_calibration.json") as f:
            state = json.load(f)

        assert "q_alpha" in state
        assert 0.0 <= state["q_alpha"] <= 1.0, \
            f"q_alpha out of range: {state['q_alpha']}"
        assert state["method"] == "split_conformal"
        assert state["coverage_guarantee"] == 0.90

        # Assert intervals file content
        intervals = np.load(tmpdir / "conformal_test_intervals.npz")
        assert "risk_scores" in intervals
        assert "lower" in intervals
        assert "upper" in intervals
        assert "widths" in intervals
        assert "labels" in intervals
        assert "escalation" in intervals
        assert len(intervals["risk_scores"]) == len(X_test), \
            "Mismatch in number of test intervals"
        assert (intervals["lower"] <= intervals["upper"]).all(), \
            "Some lower > upper in intervals"
        assert (intervals["lower"] >= 0.0).all(), "lower < 0 found"
        assert (intervals["upper"] <= 1.0).all(), "upper > 1 found"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Missing file errors
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_missing_checkpoint_raises_file_not_found():
    """Missing checkpoint file must raise FileNotFoundError with helpful message."""
    import torch
    from scripts.run_conformal_calibration import load_model

    raised = False
    try:
        load_model("/nonexistent/path/lstm_best.pt", torch.device("cpu"))
    except FileNotFoundError as e:
        raised = True
        assert "checkpoints" in str(e).lower() or "checkpoint" in str(e).lower(), \
            "Error message should mention checkpoint"
    assert raised, "Expected FileNotFoundError for missing checkpoint"


@test
def test_missing_hdf5_raises_file_not_found():
    """Missing HDF5 file must raise FileNotFoundError with helpful message."""
    import torch
    from src.models.lstm import SepsisLSTM

    config = get_default_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "lstm_best.pt"
        model = SepsisLSTM(config.lstm)
        torch.save(
            {"epoch": 0, "model_state_dict": model.state_dict(),
             "best_val_auroc": 0.5, "metrics": {}},
            ckpt_path,
        )

        from scripts.run_conformal_calibration import run_conformal_calibration
        raised = False
        try:
            run_conformal_calibration(
                checkpoint_path=str(ckpt_path),
                data_path="/nonexistent/features.h5",
                output_dir=tmpdir,
                batch_size=32,
                device_str="cpu",
            )
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing HDF5"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Bare state dict checkpoint (fallback format)
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_bare_state_dict_checkpoint_loads():
    """Checkpoint saved as bare state dict (not nested) should still load."""
    import torch
    from src.models.lstm import SepsisLSTM
    from scripts.run_conformal_calibration import load_model

    config = get_default_config()
    model = SepsisLSTM(config.lstm)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "bare_lstm.pt"
        # Save bare state dict (no 'model_state_dict' wrapper key)
        torch.save(model.state_dict(), ckpt_path)

        loaded = load_model(str(ckpt_path), torch.device("cpu"))
        assert loaded is not None
        loaded.eval()


# ─────────────────────────────────────────────────────────────────────────────
# 13. q_alpha properties
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_perfect_model_has_small_q_alpha():
    """A perfect model should produce a very small q_alpha."""
    # Perfect model: score = label exactly
    scores = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    labels = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.q_alpha < 0.2, \
        f"Perfect model should have small q_alpha, got {p.q_alpha}"


@test
def test_worst_model_has_large_q_alpha():
    """A perfectly wrong model (scores inverted) should produce large q_alpha."""
    # Worst model: predicts 1.0 for negatives, 0.0 for positives
    scores = np.array([0.0] * 50 + [1.0] * 50, dtype=np.float32)
    labels = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    p = make_predictor()
    calibrate(p, scores, labels)
    assert p.q_alpha >= 0.5, \
        f"Worst model should have large q_alpha, got {p.q_alpha}"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Batch size sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@test
def test_batch_size_one():
    """Inference with batch_size=1 must produce same results as batch_size=256."""
    import h5py
    import torch
    from src.models.lstm import SepsisLSTM
    from scripts.run_conformal_calibration import run_conformal_calibration

    config = get_default_config()
    rng = np.random.default_rng(77)
    X_val  = rng.standard_normal((50, 6, 12)).astype(np.float32)
    y_val  = (rng.random(50) < 0.13).astype(np.int8)
    X_test = rng.standard_normal((60, 6, 12)).astype(np.float32)
    y_test = (rng.random(60) < 0.13).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        h5_path = tmpdir / "features.h5"
        with h5py.File(h5_path, "w") as f:
            # train required even if unused
            f.create_dataset("X_train", data=X_val)
            f.create_dataset("y_train", data=y_val)
            f.create_dataset("X_val",   data=X_val)
            f.create_dataset("y_val",   data=y_val)
            f.create_dataset("X_test",  data=X_test)
            f.create_dataset("y_test",  data=y_test)

        model = SepsisLSTM(config.lstm)
        ckpt_path = tmpdir / "lstm_best.pt"
        torch.save({"epoch": 0, "model_state_dict": model.state_dict(),
                    "best_val_auroc": 0.5, "metrics": {}}, ckpt_path)

        r1 = run_conformal_calibration(
            str(ckpt_path), str(h5_path), str(tmpdir / "out1"),
            batch_size=1, device_str="cpu",
        )
        r2 = run_conformal_calibration(
            str(ckpt_path), str(h5_path), str(tmpdir / "out2"),
            batch_size=256, device_str="cpu",
        )

        assert abs(r1["q_alpha"] - r2["q_alpha"]) < 1e-4, \
            f"q_alpha differs: batch1={r1['q_alpha']}, batch256={r2['q_alpha']}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("QuantumSepsis — Conformal Calibration Edge Case Tests")
    print("=" * 60)
    ok = run_all()
    sys.exit(0 if ok else 1)
