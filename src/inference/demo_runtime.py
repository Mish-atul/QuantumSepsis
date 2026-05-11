"""
Runtime inference service for Streamlit demo.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.agents.orchestrator import ConfidenceGatedOrchestrator
from src.agents.red_team import RedTeamAgent
from src.baselines.xgboost_baseline import XGBoostBaseline
from src.config import get_default_config
from src.models.conformal import ConformalSepsisPredictor
from src.models.lstm import SepsisLSTM


@dataclass
class RuntimeStatus:
    backend_mode: str
    warnings: List[str]
    artifact_paths: Dict[str, str]


class DemoInferenceRuntime:
    """Single-window runtime for demo UI with graceful fallbacks."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = Path(root_dir or Path(__file__).resolve().parents[2])
        self.config = get_default_config()
        self.feature_names = list(self.config.data.feature_names)
        self.device = torch.device("cpu")

        self.lstm = self._load_lstm()
        self.xgb_model = self._load_xgb()
        self.conformal = self._load_conformal()
        self.norm_means, self.norm_stds = self._load_norm_stats()
        self.red_team = RedTeamAgent(self.config.red_team)
        self.orchestrator = ConfidenceGatedOrchestrator(self.config.orchestrator)

        self.lstm_weight = 0.3
        self.xgb_weight = 0.7
        self.xgb_fe = XGBoostBaseline(self.config)

        backend_mode = "ensemble" if self.xgb_model is not None else "lstm_only"
        self.status = RuntimeStatus(
            backend_mode=backend_mode,
            warnings=self._warnings,
            artifact_paths=self._artifact_paths,
        )

    def predict_one(self, raw_window: np.ndarray) -> Dict:
        """Predict one 6x12 raw vitals window."""
        self._validate_window(raw_window)
        window_norm = self._normalize(raw_window)

        lstm_score = self._predict_lstm(window_norm)
        xgb_score = None
        if self.xgb_model is not None:
            xgb_score = self._predict_xgb(window_norm)
            risk_score = self.lstm_weight * lstm_score + self.xgb_weight * xgb_score
        else:
            risk_score = lstm_score

        _, lower, upper, _ = self.conformal.predict(float(risk_score))
        confidence = max(0.0, min(1.0, 1.0 - (upper - lower)))

        red_team_result = self.red_team.evaluate(raw_window)
        decision = self.orchestrator.decide(
            risk_score=float(risk_score),
            conformal_lower=float(lower),
            conformal_upper=float(upper),
            red_team=red_team_result,
        )

        return {
            "risk_score": float(risk_score),
            "lstm_score": float(lstm_score),
            "xgb_score": None if xgb_score is None else float(xgb_score),
            "conformal_lower": float(lower),
            "conformal_upper": float(upper),
            "confidence": float(confidence),
            "red_team": red_team_result,
            "decision": decision,
            "status": self.status,
        }

    def _validate_window(self, raw_window: np.ndarray) -> None:
        if raw_window.shape != (6, 12):
            raise ValueError(f"Expected raw window shape (6, 12), got {raw_window.shape}")
        if np.isnan(raw_window).any():
            raise ValueError("Input window contains NaN values.")

    def _normalize(self, raw_window: np.ndarray) -> np.ndarray:
        return (raw_window - self.norm_means) / (self.norm_stds + 1e-8)

    @torch.no_grad()
    def _predict_lstm(self, window_norm: np.ndarray) -> float:
        window_tensor = torch.as_tensor(window_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        outputs = self.lstm(window_tensor)
        return float(outputs["risk_score"].item())

    def _predict_xgb(self, window_norm: np.ndarray) -> float:
        arr = np.asarray(window_norm, dtype=np.float32)[None, :, :]
        features = self.xgb_fe.engineer_features(arr)
        return float(self.xgb_model.predict_proba(features)[:, 1][0])

    def _resolve_path(self, candidates: List[str]) -> Path | None:
        for candidate in candidates:
            path = self.root_dir / candidate
            if path.exists():
                return path
        return None

    def _load_lstm(self) -> SepsisLSTM:
        self._warnings: List[str] = []
        self._artifact_paths: Dict[str, str] = {}
        lstm_path = self._resolve_path([
            "checkpoints/lstm_v1_improved_best.pt",
            "checkpoints/lstm_best.pt",
            "lstm_v1_improved_best.pt.zip",
        ])
        if lstm_path is None:
            raise FileNotFoundError("LSTM checkpoint not found in checkpoints/ directory.")

        self._artifact_paths["lstm"] = str(lstm_path)
        model = SepsisLSTM(self.config.lstm).to(self.device)
        checkpoint = torch.load(lstm_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _load_xgb(self):
        xgb_path = self._resolve_path(["checkpoints/xgboost_baseline.pkl"])
        if xgb_path is None:
            self._warnings.append("XGBoost artifact missing: falling back to LSTM-only backend.")
            return None

        self._artifact_paths["xgboost"] = str(xgb_path)
        with xgb_path.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "model" in data:
            return data["model"]
        return data

    def _load_conformal(self) -> ConformalSepsisPredictor:
        conformal_path = self._resolve_path([
            "data/processed/ensemble_conformal_calibration.json",
            "data/processed/conformal_calibration.json",
        ])
        predictor = ConformalSepsisPredictor(self.config.conformal)

        if conformal_path is None:
            predictor.q_alpha = 0.3923025131
            predictor.calibrated = True
            self._warnings.append("Conformal calibration JSON missing: using demo q_alpha fallback.")
            return predictor

        self._artifact_paths["conformal"] = str(conformal_path)
        state = json.loads(conformal_path.read_text(encoding="utf-8"))
        predictor.q_alpha = float(state.get("q_alpha", 0.3923025131))
        predictor.calibrated = True
        return predictor

    def _load_norm_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        norm_path = self._resolve_path(["data/processed/normalization_stats.json"])
        if norm_path is None:
            self._warnings.append("Normalization stats missing: using identity normalization.")
            means = np.zeros(len(self.feature_names), dtype=np.float32)
            stds = np.ones(len(self.feature_names), dtype=np.float32)
            return means, stds

        self._artifact_paths["norm_stats"] = str(norm_path)
        state = json.loads(norm_path.read_text(encoding="utf-8"))
        means = np.array([state["train_mean"][f] for f in self.feature_names], dtype=np.float32)
        stds = np.array([state["train_std"][f] for f in self.feature_names], dtype=np.float32)
        stds[stds == 0.0] = 1.0
        return means, stds
