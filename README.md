# QuantumSepsis Shield
# Adversarially-Safe Quantum-Classical System for Early Sepsis Detection

## Quick Start

```bash
# Create environment
conda create -n quantumsepsis python=3.10 -y
conda activate quantumsepsis

# Install dependencies
pip install -r requirements.txt

# Run data pipeline (requires MIMIC-IV access)
python -m src.data.cohort_extraction --config config.yaml
python -m src.data.feature_extraction --config config.yaml
python -m src.data.preprocessing --config config.yaml
python -m src.data.windowing --config config.yaml

# Train LSTM baseline
python -m src.training.train_lstm --config config.yaml

# Run baselines
python -m src.baselines.xgboost_baseline --config config.yaml
python -m src.baselines.sofa_baseline --config config.yaml
```

## Project Structure

```
QuantumSepsis/
├── files/              # Documentation (architecture, dataset, novelty, roadmap)
├── src/                # Source code
│   ├── data/           # Data pipeline modules
│   ├── models/         # ML models (LSTM, quantum kernel, conformal)
│   ├── agents/         # Safety agents (Red Team, Orchestrator, Outcome Learner)
│   ├── training/       # Training loops
│   ├── evaluation/     # Metrics and evaluation
│   └── baselines/      # Baseline comparisons
├── data/               # Data directory (not committed)
│   ├── raw/            # Raw MIMIC-IV files
│   └── processed/      # Processed features
├── checkpoints/        # Model checkpoints
├── logs/               # Training logs
└── requirements.txt
```

## Team
- Yash Gautam — Data Pipeline + Training
- Atul Kumar Mishra — Model Architecture + Quantum Kernel
- Tanishk Viraj Bhanage — Safety Agents + Evaluation

## Dataset
MIMIC-IV v3.1 (PhysioNet credentialed access required)

## License
Research use only — MIMIC-IV data usage agreement applies.
