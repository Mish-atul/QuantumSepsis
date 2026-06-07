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
    """Data pipeline configuration for Sepsis detection."""
    # MIMIC-IV paths
    mimic_raw_dir: str = "data/raw/mimiciv/3.1"
    processed_dir: str = "data/processed/sepsis"

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
class FouDataConfig:
    """Data pipeline configuration for FOU detection."""
    # MIMIC-IV paths
    mimic_raw_dir: str = "data/raw/mimiciv/3.1"
    processed_dir: str = "data/processed/fou"

    # BigQuery settings
    use_bigquery: bool = False
    bigquery_project: str = "physionet-data"
    bigquery_dataset_hosp: str = "mimiciv_v3_1_hosp"
    bigquery_dataset_icu: str = "mimiciv_v3_1_icu"

    # Feature extraction - reuse sepsis vitals
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

    # FOU-specific lab item IDs (12 reused + 15 new)
    lab_item_ids: Dict[str, int] = field(default_factory=lambda: {
        # Reused from sepsis
        "lactate": 50813,
        "wbc": 51301,
        "creatinine": 50912,
        "platelets": 51265,
        # FOU-specific inflammatory markers
        "crp": 50889,
        "esr": 51288,
        "procalcitonin": 50963,
        "ferritin": 50896,
        "ldh": 50954,
        "albumin": 50862,
    })

    # FOU cohort criteria
    min_icu_stay_days: int = 7
    fever_threshold_celsius: float = 38.3
    min_fever_episodes: int = 3

    # Preprocessing (longer windows for FOU)
    bin_size_hours: int = 1
    forward_fill_limit_hours: int = 6  # Longer for FOU
    window_size_hours: int = 24  # 24-hour windows
    window_stride_hours: int = 6  # 6-hour stride
    prediction_horizon_hours: int = 48  # Predict 48 hours ahead

    # Feature names (27 total: 12 reused + 15 FOU-specific)
    feature_names: List[str] = field(default_factory=lambda: [
        # 12 reused from sepsis
        "heart_rate", "sbp", "dbp", "map", "temperature",
        "resp_rate", "spo2", "gcs_total",
        "lactate", "wbc", "creatinine", "platelets",
        # 15 FOU-specific
        "temp_max_24h", "temp_variability", "fever_duration_hours",
        "crp", "esr", "procalcitonin", "ferritin", "ldh", "albumin",
        "antibiotic_days", "culture_negative_count", "immunosuppressed",
        "weight_loss", "night_sweats_proxy", "rash_documented"
    ])

    n_features: int = 27

    # Split configuration (same as sepsis)
    train_year_groups: List[str] = field(default_factory=lambda: [
        "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"
    ])
    test_year_groups: List[str] = field(default_factory=lambda: [
        "2020 - 2022"
    ])
    val_fraction: float = 0.15


@dataclass
class LSTMConfig:
    """LSTM model configuration for Sepsis detection."""
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
class FouLSTMConfig:
    """LSTM model configuration for FOU detection."""
    input_size: int = 27       # Number of features (12 reused + 15 FOU-specific)
    seq_len: int = 24          # Window size (24 hours)
    hidden_dim: int = 128      # LSTM hidden dimension
    n_layers: int = 2          # Number of LSTM layers
    bidirectional: bool = True
    dropout: float = 0.3
    attention_dim: int = 64    # Temporal attention dimension
    fc1_dim: int = 64          # First FC layer output
    embedding_dim: int = 16    # Latent embedding (→ quantum input)

    # Output
    n_classes: int = 4         # Multi-class: No FOU, Infectious, Non-infectious, Undiagnosed


@dataclass
class TrainingConfig:
    """Training configuration for Sepsis detection."""
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
    checkpoint_dir: str = "checkpoints/sepsis"
    log_dir: str = "logs/sepsis"

    # W&B
    use_wandb: bool = True
    wandb_project: str = "quantumsepsis-shield"
    wandb_entity: Optional[str] = None


@dataclass
class FouTrainingConfig:
    """Training configuration for FOU detection."""
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
    early_stopping_metric: str = "val_macro_f1"  # Multi-class metric
    gradient_clip_norm: float = 1.0

    # Loss - Multi-class Focal Loss
    focal_alpha: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.3, 0.2])  # [No FOU, Infectious, Non-infectious, Undiagnosed]
    focal_gamma: float = 2.0

    # Reproducibility
    seed: int = 42

    # Paths
    checkpoint_dir: str = "checkpoints/fou"
    log_dir: str = "logs/fou"

    # W&B
    use_wandb: bool = True
    wandb_project: str = "quantumsepsis-fou"
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
    """Red Team Agent configuration for Sepsis detection."""
    # Tripwire thresholds
    temp_low: float = 36.0          # °C
    temp_high: float = 38.3         # °C
    hr_threshold: float = 90.0     # bpm
    hr_trend_threshold: float = 5.0 # bpm/hr
    rr_threshold: float = 20.0     # breaths/min
    map_threshold: float = 70.0    # mmHg
    gcs_threshold: float = 12.0    # GCS score (moderate impairment threshold)

    # Escalation
    critical_tripwire_count: int = 2  # >= 2 tripwires → CRITICAL


@dataclass
class FouRedTeamConfig:
    """Red Team Agent configuration for FOU detection."""
    # FOU-specific tripwire thresholds
    persistent_fever_temp: float = 39.0      # °C
    persistent_fever_hours: float = 72.0     # hours

    # qSOFA criteria for sepsis rule-out
    qsofa_threshold: int = 2                 # qSOFA ≥ 2 → rule out sepsis first
    qsofa_rr_threshold: float = 22.0         # breaths/min
    qsofa_sbp_threshold: float = 100.0       # mmHg
    qsofa_gcs_threshold: float = 15.0        # GCS < 15

    # Neutropenia
    anc_threshold: float = 500.0             # ANC < 500 → high infection risk

    # Hemodynamic instability
    map_threshold: float = 65.0              # mmHg

    # Altered mental status
    gcs_threshold: float = 13.0              # GCS < 13

    # Escalation
    critical_tripwire_count: int = 2         # >= 2 tripwires → CRITICAL


@dataclass
class ConformalConfig:
    """Conformal prediction configuration for Sepsis detection."""
    method: str = "split"           # split conformal
    alpha: float = 0.10             # 90% coverage guarantee
    calibration_fraction: float = 0.20  # 20% of training positives
    escalation_width_threshold: float = 0.40  # Width > 0.4 → escalate


@dataclass
class FouConformalConfig:
    """Conformal prediction configuration for FOU detection."""
    method: str = "aps"             # Adaptive Prediction Sets for multi-class
    alpha: float = 0.10             # 90% coverage guarantee
    calibration_fraction: float = 0.20  # 20% of training samples
    max_set_size: int = 3           # Maximum prediction set size


@dataclass
class OrchestratorConfig:
    """Intervention Orchestrator configuration for Sepsis detection."""
    # Risk thresholds
    watch_threshold: float = 0.3
    amber_threshold: float = 0.6

    # Confidence-gated fast-tracking
    high_confidence: float = 0.80
    low_confidence: float = 0.50


@dataclass
class UnifiedOrchestratorConfig:
    """Unified Orchestrator configuration for multi-condition detection."""
    # Sepsis thresholds (more acute, takes priority)
    sepsis_watch_threshold: float = 0.3
    sepsis_amber_threshold: float = 0.6

    # FOU thresholds
    fou_watch_threshold: float = 0.5
    fou_amber_threshold: float = 0.7

    # Priority logic
    sepsis_priority_threshold: float = 0.3  # If sepsis risk > 0.3, prioritize sepsis
    fou_activation_threshold: float = 0.5   # If sepsis < 0.3 and FOU > 0.5, activate FOU workup

    # Confidence-gated fast-tracking
    high_confidence: float = 0.80
    low_confidence: float = 0.50


@dataclass
class Config:
    """Master configuration for Sepsis detection."""
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


@dataclass
class FouConfig:
    """Master configuration for FOU detection."""
    data: FouDataConfig = field(default_factory=FouDataConfig)
    lstm: FouLSTMConfig = field(default_factory=FouLSTMConfig)
    training: FouTrainingConfig = field(default_factory=FouTrainingConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)  # Reuse quantum config
    red_team: FouRedTeamConfig = field(default_factory=FouRedTeamConfig)
    conformal: FouConformalConfig = field(default_factory=FouConformalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FouConfig":
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


@dataclass
class UnifiedConfig:
    """Unified configuration for multi-condition detection (Sepsis + FOU)."""
    sepsis: Config = field(default_factory=Config)
    fou: FouConfig = field(default_factory=FouConfig)
    orchestrator: UnifiedOrchestratorConfig = field(default_factory=UnifiedOrchestratorConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "UnifiedConfig":
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


# Default configuration instances
def get_default_config() -> Config:
    """Get default configuration for Sepsis detection."""
    return Config()


def get_default_fou_config() -> FouConfig:
    """Get default configuration for FOU detection."""
    return FouConfig()


def get_default_unified_config() -> UnifiedConfig:
    """Get default unified configuration for multi-condition detection."""
    return UnifiedConfig()


if __name__ == "__main__":
    # Test configurations
    print("=== Sepsis Configuration ===")
    config = get_default_config()
    print(f"Features: {config.data.feature_names}")
    print(f"LSTM hidden dim: {config.lstm.hidden_dim}")
    print(f"Quantum qubits: {config.quantum.n_qubits}")
    print(f"Focal alpha (pos): {config.training.focal_alpha_pos}")
    print(f"Red Team temp threshold: {config.red_team.temp_low}–{config.red_team.temp_high}°C")

    print("\n=== FOU Configuration ===")
    fou_config = get_default_fou_config()
    print(f"Features: {len(fou_config.data.feature_names)} total")
    print(f"Window size: {fou_config.data.window_size_hours} hours")
    print(f"LSTM input size: {fou_config.lstm.input_size}")
    print(f"LSTM n_classes: {fou_config.lstm.n_classes}")
    print(f"Focal alpha: {fou_config.training.focal_alpha}")

    print("\n=== Unified Configuration ===")
    unified_config = get_default_unified_config()
    print(f"Sepsis priority threshold: {unified_config.orchestrator.sepsis_priority_threshold}")
    print(f"FOU activation threshold: {unified_config.orchestrator.fou_activation_threshold}")

    # Save default configs
    config.to_yaml("config_sepsis_default.yaml")
    fou_config.to_yaml("config_fou_default.yaml")
    unified_config.to_yaml("config_unified_default.yaml")
    print("\nDefault configs saved.")
