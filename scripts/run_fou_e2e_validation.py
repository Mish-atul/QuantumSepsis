"""
FOU End-to-End Validation
==========================
Complete pipeline validation for FOU detection:
- LSTM inference
- Conformal prediction
- Red Team assessment
- Unified orchestrator decision

Usage:
    python scripts/run_fou_e2e_validation.py --model checkpoints/fou/fou_lstm_best.pt --data data/processed/fou/fou_features.h5
"""

import sys
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm_fou import FouLSTM
from src.models.conformal_fou import MultiClassConformalPredictor
from src.agents.red_team_fou import FouRedTeamAgent
from src.agents.orchestrator_unified import UnifiedOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="FOU E2E Validation")
    parser.add_argument('--model', type=str, required=True, help='Path to trained FOU LSTM')
    parser.add_argument('--data', type=str, required=True, help='Path to FOU features HDF5')
    parser.add_argument('--conformal', type=str, default='data/processed/fou/fou_conformal_calibration.json', help='Conformal calibration')
    parser.add_argument('--output-dir', type=str, default='data/processed/fou', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max test samples to process')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = FouLSTM(
        input_size=config.lstm.input_size,
        seq_len=config.lstm.seq_len,
        hidden_dim=config.lstm.hidden_dim,
        n_layers=config.lstm.n_layers,
        bidirectional=config.lstm.bidirectional,
        dropout=config.lstm.dropout,
        attention_dim=config.lstm.attention_dim,
        fc1_dim=config.lstm.fc1_dim,
        embedding_dim=config.lstm.embedding_dim,
        n_classes=config.lstm.n_classes
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load conformal calibration
    logger.info(f"Loading conformal calibration from {args.conformal}...")
    with open(args.conformal, 'r') as f:
        conformal_config = json.load(f)

    conformal = MultiClassConformalPredictor(
        alpha=conformal_config['alpha'],
        n_classes=4
    )
    conformal.q_alpha = conformal_config['q_alpha']

    # Initialize agents
    red_team = FouRedTeamAgent()
    orchestrator = UnifiedOrchestrator()

    # Load test data
    logger.info(f"Loading test data from {args.data}...")
    with h5py.File(args.data, 'r') as f:
        X_test = f['X_test'][:args.max_samples]
        y_test = f['y_test'][:args.max_samples]

    logger.info(f"Test samples: {len(X_test)}")

    # Create dataloader
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Get all predictions first
    logger.info("Running LSTM inference...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="LSTM inference"):
            X_batch = X_batch.to(device)
            _, probs, _, _ = model(X_batch)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)

    # Get conformal prediction sets
    logger.info("Computing conformal prediction sets...")
    prediction_sets, set_sizes = conformal.predict(all_probs)

    # E2E validation
    logger.info("Running E2E validation...")

    decisions = []
    alert_level_counts = {"WATCH": 0, "AMBER": 0, "CRITICAL": 0, "FAST-TRACK": 0}
    primary_condition_counts = {"sepsis": 0, "fou": 0, "both": 0, "neither": 0}
    red_team_override_count = 0

    feature_names = config.data.feature_names

    for i in tqdm(range(len(X_test)), desc="E2E validation"):
        # Get current window features
        window = X_test[i]  # (24, 27)

        # Extract latest features (last time step)
        latest_features = window[-1, :]

        # Create feature dict
        features = {name: float(latest_features[j]) for j, name in enumerate(feature_names)}

        # FOU probabilities
        fou_probs = all_probs[i]
        fou_pred_set = prediction_sets[i]

        # Red Team assessment
        red_team_assessment = red_team.assess(features, window)

        # Unified orchestrator decision (assuming low sepsis risk for FOU-focused validation)
        sepsis_risk = 0.1  # Placeholder - in real scenario would come from sepsis model

        decision = orchestrator.decide(
            sepsis_risk=sepsis_risk,
            fou_probabilities=fou_probs,
            sepsis_conformal=None,
            fou_conformal_set=fou_pred_set,
            red_team_sepsis=None,
            red_team_fou=red_team_assessment,
            features=features
        )

        decisions.append(decision)

        # Count alert levels
        alert_level_counts[decision.alert_level.value] += 1
        primary_condition_counts[decision.primary_condition] += 1

        if decision.red_team_override:
            red_team_override_count += 1

    # Compute metrics
    logger.info("\n=== FOU E2E Validation Results ===")

    # Conformal metrics
    conformal_metrics = conformal.evaluate(all_probs, all_labels)
    logger.info(f"\nConformal Prediction:")
    logger.info(f"  Coverage: {conformal_metrics['coverage']:.4f}")
    logger.info(f"  Avg set size: {conformal_metrics['avg_set_size']:.2f}")
    logger.info(f"  Singleton rate: {conformal_metrics['singleton_rate']:.4f}")

    # Alert level distribution
    logger.info(f"\nAlert Level Distribution:")
    for level, count in alert_level_counts.items():
        pct = 100 * count / len(decisions)
        logger.info(f"  {level}: {count} ({pct:.1f}%)")

    # Primary condition distribution
    logger.info(f"\nPrimary Condition Distribution:")
    for condition, count in primary_condition_counts.items():
        pct = 100 * count / len(decisions)
        logger.info(f"  {condition}: {count} ({pct:.1f}%)")

    # Red Team overrides
    override_rate = 100 * red_team_override_count / len(decisions)
    logger.info(f"\nRed Team Overrides: {red_team_override_count} ({override_rate:.1f}%)")

    # Save results
    results = {
        "n_samples": len(X_test),
        "conformal_metrics": {k: float(v) for k, v in conformal_metrics.items()},
        "alert_level_distribution": alert_level_counts,
        "primary_condition_distribution": primary_condition_counts,
        "red_team_override_count": red_team_override_count,
        "red_team_override_rate": override_rate
    }

    results_path = output_dir / "fou_e2e_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Sample decisions
    logger.info("\n=== Sample Decisions ===")
    for i in range(min(5, len(decisions))):
        decision = decisions[i]
        true_label = all_labels[i]
        class_names = ["No FOU", "Infectious", "Non-infectious", "Undiagnosed"]

        logger.info(f"\nSample {i+1}:")
        logger.info(f"  True label: {class_names[true_label]}")
        logger.info(f"  FOU probabilities: {decision.fou_probabilities}")
        logger.info(f"  Predicted class: {class_names[decision.fou_predicted_class]}")
        logger.info(f"  Conformal set: {[class_names[c] for c in decision.fou_conformal_set]}")
        logger.info(f"  Primary condition: {decision.primary_condition}")
        logger.info(f"  Alert level: {decision.alert_level.value}")
        logger.info(f"  Actions: {decision.actions[:2]}")

    logger.info("\n✓ FOU E2E validation complete")


if __name__ == "__main__":
    main()
