"""
QuantumSepsis Shield — Global Configuration
============================================
Central configuration for all pipeline components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # MIMIC-IV paths
    mimic_raw_dir: str = "data/raw/mimiciv/3.1"
    processed_dir: str = "data/processed"
    
    # BigQuery settings (alternative to local files)
    use_bigquery: bool = False
    bigquery_project: str = "physionet-data"
    bigquery_dataset_hosp: str = "mimiciv_v3_1_hosp"
    bigquery_dataset_icu: str = "mimiciv_v3_1_icu"
    
    # Feature extraction
    vitals_item_ids: Dict[str, List[int]] = field(default_factory=lambda: {
        "heart_rate": [211, 220045],
        "sbp": [51, 442, 455, 6701, 220179, 220050],
        "dbp": [8368, 8440, 8441, 8555, 220180, 220051],
        "map": [52, 456, 6702, 220052, 220181],
        "temperature": [223762, 226329],
        "resp_rate": [615, 618, 220210, 224690],
        "spo2": [646, 220277],
        "gcs_total": [198, 226755, 227013],
    })
    
    lab_item_ids: Dict[str, int] = field(default_factory=lambda: {
        "lactate": 50813,
        "wbc": 51301,
        "creatinine": 50912,
        "bilirubin": 50885,
        "platelets": 51265,
    })
    
    # Preprocessing
    bin_size_hours: int = 1
    forward_fill_limit_hours: int = 2
    window_size_hours: int = 6
    window_stride_hours: int = 1
    prediction_horizon_hours: int = 4  # Predict 4 hours before onset
    
    # Feature names (order matters — matches tensor columns)
    feature_names: List[str] = field(default_factory=lambda: [
        "heart_rate", "sbp", "dbp", "map", "temperature",
        "resp_rate", "spo2", "gcs_total",
        "lactate", "wbc", "creatinine", "platelets"
    ])
    
    n_features: int = 12
    
    # Split configuration
    train_year_groups: List[str] = field(default_factory=lambda: [
        "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"
    ])
    test_year_groups: List[str] = field(default_factory=lambda: [
        "2020 - 2022"
    ])
    val_fraction: float = 0.15  # From training set


@dataclass
class LSTMConfig:
    """LSTM model configuration."""
    input_size: int = 12       # Number of features
    seq_len: int = 6           # Window size (hours)
    hidden_dim: int = 128      # LSTM hidden dimension
    n_layers: int = 2          # Number of LSTM layers
    bidirectional: bool = True
    dropout: float = 0.3
    attention_dim: int = 64    # Temporal attention dimension
    fc1_dim: int = 64          # First FC layer output
    embedding_dim: int = 16    # Latent embedding (→ quantum input)
    
    # Output
    n_classes: int = 1         # Binary classification


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine"
    scheduler_t_max: int = 50
    
    # Training
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auroc"
    gradient_clip_norm: float = 1.0
    
    # Loss
    focal_alpha_pos: float = 0.9    # Weight for positive class (FN penalty)
    focal_alpha_neg: float = 0.1    # Weight for negative class
    focal_gamma: float = 2.0        # Focusing parameter
    
    # Reproducibility
    seed: int = 42
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # W&B
    use_wandb: bool = True
    wandb_project: str = "quantumsepsis-shield"
    wandb_entity: Optional[str] = None


@dataclass
class QuantumConfig:
    """Quantum kernel configuration."""
    n_qubits: int = 8              # Number of qubits
    feature_map: str = "ZZFeatureMap"
    entanglement: str = "linear"
    reps: int = 2                   # Feature map repetitions
    pca_components: int = 8         # PCA reduction from embedding_dim
    backend: str = "aer_simulator"  # or "ibm_quantum"
    shots: int = 1024               # Measurement shots
    
    # IBM Quantum (for production)
    ibm_token: Optional[str] = None
    ibm_backend: str = "ibm_brisbane"


@dataclass
class RedTeamConfig:
    """Red Team Agent configuration."""
    # Tripwire thresholds
    temp_low: float = 36.0          # °C
    temp_high: float = 38.3         # °C
    hr_threshold: float = 90.0     # bpm
    hr_trend_threshold: float = 5.0 # bpm/hr
    rr_threshold: float = 20.0     # breaths/min
    map_threshold: float = 70.0    # mmHg
    gcs_threshold: float = 8.0     # GCS score (adjusted for ICU population, was 14.0)
    
    # Escalation
    critical_tripwire_count: int = 2  # >= 2 tripwires → CRITICAL


@dataclass
class ConformalConfig:
    """Conformal prediction configuration."""
    method: str = "split"           # split conformal
    alpha: float = 0.10             # 90% coverage guarantee
    calibration_fraction: float = 0.20  # 20% of training positives
    escalation_width_threshold: float = 0.40  # Width > 0.4 → escalate


@dataclass
class OrchestratorConfig:
    """Intervention Orchestrator configuration."""
    # Risk thresholds
    watch_threshold: float = 0.3
    amber_threshold: float = 0.6
    
    # Confidence-gated fast-tracking
    high_confidence: float = 0.80
    low_confidence: float = 0.50


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    red_team: RedTeamConfig = field(default_factory=RedTeamConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        
        config = cls()
        if raw:
            for section_name, section_data in raw.items():
                if hasattr(config, section_name) and isinstance(section_data, dict):
                    section = getattr(config, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration instance
def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print(f"Features: {config.data.feature_names}")
    print(f"LSTM hidden dim: {config.lstm.hidden_dim}")
    print(f"Quantum qubits: {config.quantum.n_qubits}")
    print(f"Focal alpha (pos): {config.training.focal_alpha_pos}")
    print(f"Red Team temp threshold: {config.red_team.temp_low}–{config.red_team.temp_high}°C")
    
    # Save default config
    config.to_yaml("config_default.yaml")
    print("\nDefault config saved to config_default.yaml")
