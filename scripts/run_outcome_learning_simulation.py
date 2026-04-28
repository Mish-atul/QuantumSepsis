"""
QuantumSepsis Shield — Outcome Learning Agent Simulation
=========================================================

Simulates the OutcomeLearningAgent on real test-set decisions to validate:
  1. Adaptive threshold updates per ICU unit
  2. Near-miss detection (Red Team caught what model missed)
  3. Loss multiplier escalation for near-miss patient profiles
  4. Sensitivity tracking before vs. after threshold adaptation

Input:
  data/processed/e2e_decisions.npz        ← decisions from run_e2e_validation.py
  data/processed/e2e_validation_results.json

Output:
  data/processed/outcome_learning_results.json   ← per-unit threshold state + stats
  data/processed/near_miss_weights.json          ← stay_id → loss_multiplier

Usage:
    python3 scripts/run_outcome_learning_simulation.py
    python3 scripts/run_outcome_learning_simulation.py --synthetic
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.outcome_learner import OutcomeLearningAgent, CaseOutcome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ICU units used in simulation (mirrors real MIMIC-IV ICU types)
ICU_UNITS = ["MICU", "SICU", "CVICU", "CCU", "NICU"]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def alert_label_to_str(label: int) -> str:
    """Map numeric alert label back to string."""
    return {0: "WATCH", 1: "AMBER", 2: "CRITICAL"}.get(int(label), "WATCH")


def load_decisions(decisions_npz_path: str) -> dict:
    """Load per-window decisions from run_e2e_validation.py output."""
    path = Path(decisions_npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Decisions NPZ not found: {path}\n"
            "Run first:\n"
            "  python3 scripts/run_e2e_validation.py"
        )
    data = np.load(path)
    logger.info("Loaded decisions: %d windows from %s", len(data["true_labels"]), path)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build CaseOutcome records from e2e decisions
# ══════════════════════════════════════════════════════════════════════════════

def build_case_outcomes(
    decisions_data: dict,
    icu_units: list,
    rng: np.random.Generator,
) -> list:
    """Convert per-window decision arrays into CaseOutcome records.

    In production these would come from the EHR 72h post-alert.
    Here we simulate using the ground-truth labels from the test set.

    Each window is treated as a separate case assigned to a random ICU unit.
    """
    risk_scores   = decisions_data["risk_scores"]
    alert_labels  = decisions_data["alert_labels"]
    true_labels   = decisions_data["true_labels"]
    confidences   = decisions_data["confidences"]
    red_team_lvls = decisions_data["red_team_levels"]   # 0=WATCH 1=AMBER 2=CRITICAL

    outcomes = []
    n = len(true_labels)

    for i in range(n):
        alert_str       = alert_label_to_str(int(alert_labels[i]))
        actual_sepsis   = bool(true_labels[i] == 1)
        red_triggered   = bool(int(red_team_lvls[i]) > 0)
        n_tripwires     = int(red_team_lvls[i])
        icu_unit        = icu_units[int(rng.integers(0, len(icu_units)))]

        outcomes.append(CaseOutcome(
            stay_id           = i,
            icu_unit          = icu_unit,
            prediction_time   = f"2150-01-01T{i % 24:02d}:00:00",
            risk_score        = float(risk_scores[i]),
            confidence        = float(confidences[i]),
            alert_level       = alert_str,
            red_team_triggered= red_triggered,
            n_tripwires       = n_tripwires,
            actual_sepsis     = actual_sepsis,
            time_to_onset_hours= float(rng.uniform(0, 4)) if actual_sepsis else None,
        ))

    logger.info("Built %d CaseOutcome records across %d ICU units", n, len(icu_units))
    return outcomes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run OutcomeLearningAgent on all cases
# ══════════════════════════════════════════════════════════════════════════════

def run_outcome_learning(outcomes: list, min_cases_for_update: int = 20) -> dict:
    """Feed all outcomes into OutcomeLearningAgent and collect feedback.

    Returns:
        dict with agent state summary, all feedback events, near-miss profiles
    """
    agent = OutcomeLearningAgent(
        learning_rate         = 0.05,
        target_sensitivity    = 0.95,
        min_cases_for_update  = min_cases_for_update,
        update_frequency_hours= 168,
    )

    all_feedback       = []
    false_negatives    = []
    near_misses        = []
    threshold_updates  = []

    logger.info("Running OutcomeLearningAgent on %d cases...", len(outcomes))

    for outcome in outcomes:
        feedback = agent.record_outcome(outcome)

        if feedback:
            all_feedback.append({
                "stay_id" : outcome.stay_id,
                "icu_unit": outcome.icu_unit,
                "feedback": feedback,
            })

        if outcome.was_false_negative:
            false_negatives.append({
                "stay_id"    : outcome.stay_id,
                "icu_unit"   : outcome.icu_unit,
                "risk_score" : outcome.risk_score,
                "alert_level": outcome.alert_level,
            })

        if outcome.was_near_miss:
            near_misses.append({
                "stay_id"   : outcome.stay_id,
                "icu_unit"  : outcome.icu_unit,
                "risk_score": outcome.risk_score,
                "n_tripwires": outcome.n_tripwires,
            })

        if feedback and "threshold_update" in feedback:
            threshold_updates.append({
                "stay_id"  : outcome.stay_id,
                "icu_unit" : outcome.icu_unit,
                "update"   : feedback["threshold_update"],
            })

    logger.info("Feedback events    : %d", len(all_feedback))
    logger.info("False negatives    : %d", len(false_negatives))
    logger.info("Near-misses        : %d", len(near_misses))
    logger.info("Threshold updates  : %d", len(threshold_updates))

    # Per-unit final states
    unit_states = {}
    for unit in agent.unit_states:
        state = agent.unit_states[unit]
        tp    = state.true_positives
        fn    = state.false_negatives
        sens  = tp / max(tp + fn, 1)
        unit_states[unit] = {
            "total_cases"     : state.total_cases,
            "true_positives"  : state.true_positives,
            "false_negatives" : state.false_negatives,
            "false_positives" : state.false_positives,
            "near_misses"     : state.near_misses,
            "sensitivity"     : round(sens, 4),
            "loss_multiplier" : state.loss_multiplier,
            "watch_threshold" : round(state.watch_threshold, 4),
            "amber_threshold" : round(state.amber_threshold, 4),
        }
        logger.info(
            "  [%s] cases=%d  TP=%d  FN=%d  sens=%.4f  loss_mult=%.1fx  "
            "thresholds=WATCH%.2f/AMBER%.2f",
            unit, state.total_cases, tp, fn, sens,
            state.loss_multiplier, state.watch_threshold, state.amber_threshold,
        )

    near_miss_weights = agent.get_near_miss_weights()

    return {
        "n_cases"            : len(outcomes),
        "n_false_negatives"  : len(false_negatives),
        "n_near_misses"      : len(near_misses),
        "n_threshold_updates": len(threshold_updates),
        "unit_states"        : unit_states,
        "false_negatives"    : false_negatives,
        "near_misses"        : near_misses,
        "threshold_updates"  : threshold_updates,
        "near_miss_weights"  : {str(k): float(v) for k, v in near_miss_weights.items()},
        "agent_summary"      : agent.summary(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_outcome_learning_simulation(
    decisions_npz_path: str = "data/processed/e2e_decisions.npz",
    output_dir        : str = "data/processed",
    min_cases         : int = 20,
    seed              : int = 42,
) -> dict:
    """Full outcome learning simulation pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Step 1 — Load decisions
    logger.info("\n%s\nStep 1/3 — Loading e2e decisions\n%s", "="*60, "="*60)
    decisions_data = load_decisions(decisions_npz_path)

    # Step 2 — Build case outcomes
    logger.info("\n%s\nStep 2/3 — Building CaseOutcome records\n%s", "="*60, "="*60)
    outcomes = build_case_outcomes(decisions_data, ICU_UNITS, rng)

    # Step 3 — Run agent
    logger.info("\n%s\nStep 3/3 — Running OutcomeLearningAgent\n%s", "="*60, "="*60)
    result = run_outcome_learning(outcomes, min_cases_for_update=min_cases)

    # Save results
    json_path = output_dir / "outcome_learning_results.json"
    # Remove non-serializable agent_summary (plain string)
    save_result = {k: v for k, v in result.items() if k != "agent_summary"}
    with open(json_path, "w") as f:
        json.dump(save_result, f, indent=2)
    logger.info("\nSaved → %s", json_path)

    weights_path = output_dir / "near_miss_weights.json"
    with open(weights_path, "w") as f:
        json.dump(result["near_miss_weights"], f, indent=2)
    logger.info("Saved → %s", weights_path)

    # Print summary
    logger.info("\n%s\nOUTCOME LEARNING SIMULATION COMPLETE\n%s", "="*60, "="*60)
    logger.info("Total cases processed : %d", result["n_cases"])
    logger.info("False negatives       : %d", result["n_false_negatives"])
    logger.info("Near-misses           : %d", result["n_near_misses"])
    logger.info("Threshold updates     : %d", result["n_threshold_updates"])
    logger.info("Near-miss weights     : %d patient profiles flagged",
                len(result["near_miss_weights"]))
    logger.info("\n%s", result["agent_summary"])
    logger.info("="*60)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test(output_dir: str = "data/processed") -> None:
    """Full smoke test without requiring e2e_decisions.npz on disk."""
    logger.info("="*60)
    logger.info("SYNTHETIC SMOKE TEST — Outcome Learning Simulation")
    logger.info("="*60)

    rng = np.random.default_rng(0)
    n   = 300

    # Simulate what run_e2e_validation.py would produce
    risk_scores   = rng.uniform(0, 1, n).astype(np.float32)
    true_labels   = (rng.random(n) < 0.13).astype(np.int8)
    alert_labels  = np.where(risk_scores > 0.6, 2,
                    np.where(risk_scores > 0.3, 1, 0)).astype(np.int8)
    confidences   = np.clip(1.0 - 2*np.abs(risk_scores - 0.5), 0, 1).astype(np.float32)
    red_team_lvls = (rng.random(n) < 0.05).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        npz_path = tmpdir / "e2e_decisions.npz"
        np.savez(
            npz_path,
            risk_scores   = risk_scores,
            alert_labels  = alert_labels,
            true_labels   = true_labels,
            conformal_widths = np.full(n, 0.25, dtype=np.float32),
            confidences   = confidences,
            fast_tracked  = np.zeros(n, dtype=np.uint8),
            red_team_levels = red_team_lvls,
        )

        result = run_outcome_learning_simulation(
            decisions_npz_path = str(npz_path),
            output_dir         = str(tmpdir / "out"),
            min_cases          = 5,
        )

        assert result["n_cases"] == n
        assert (tmpdir / "out" / "outcome_learning_results.json").exists()
        assert (tmpdir / "out" / "near_miss_weights.json").exists()
        assert "unit_states" in result
        assert len(result["unit_states"]) > 0

        logger.info("\n✓ Synthetic smoke test passed!")
        logger.info("  FN=%d  NM=%d  ThresholdUpdates=%d",
                    result["n_false_negatives"],
                    result["n_near_misses"],
                    result["n_threshold_updates"])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outcome Learning Agent Simulation")
    parser.add_argument("--decisions-npz", default="data/processed/e2e_decisions.npz")
    parser.add_argument("--output-dir",    default="data/processed")
    parser.add_argument("--min-cases",     type=int, default=20,
                        help="Min resolved cases before threshold update (default: 20)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--synthetic",     action="store_true",
                        help="Run smoke test without real data")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_test(output_dir=args.output_dir)
    else:
        run_outcome_learning_simulation(
            decisions_npz_path = args.decisions_npz,
            output_dir         = args.output_dir,
            min_cases          = args.min_cases,
            seed               = args.seed,
        )
