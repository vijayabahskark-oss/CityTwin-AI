"""
risk_assessor.py — High-Level RiskAssessor Interface
=====================================================
Provides an end-to-end evaluation API that seamlessly chains:
  1. Historical CityState sequence (e.g. last 10 timesteps)
  2. LSTM temporal prediction (Unit III: TrafficPredictor)
  3. Risk feature vector construction (29-dim)
  4. Neural risk inference (Unit IV: RiskFNN)
  5. Discrete classification, factor attribution, and actionable mitigation advisories

Usage
-----
>>> from risk import RiskAssessor
>>> assessor = RiskAssessor.from_checkpoints()
>>> report = assessor.assess(recent_states)
>>> print(report.overall_risk_score, report.risk_level)
>>> print(report.recommended_actions)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.mock_city import CityState
from prediction.feature_extractor import FeatureExtractor
from prediction.predictor import TrafficPredictor
from risk.risk_calculator import (
    RiskLevel,
    classify_risk_level,
    RISK_TARGET_NAMES,
)
from risk.dataset import construct_risk_features, RISK_FEATURE_DIM
from risk.risk_model import RiskFNN


# ---------------------------------------------------------------------------
# Structured Report Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """
    Comprehensive structured risk assessment output.
    """
    overall_risk_score: float         # [0, 1]
    risk_level: str                   # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    congestion_risk: float            # [0, 1]
    safety_hazard_risk: float         # [0, 1]
    emergency_delay_risk: float       # [0, 1]
    primary_risk_factors: List[str]   # Contributing hazard factors
    recommended_actions: List[str]    # Automated control / mitigation advisories
    forecast_summary: Dict[str, float]# Summary statistics from temporal forecast

    def to_dict(self) -> Dict:
        """Convert report to plain serializable dictionary."""
        return asdict(self)

    def is_critical(self) -> bool:
        """Check if condition requires immediate intervention."""
        return self.risk_level in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)

    def __str__(self) -> str:
        factors = ", ".join(self.primary_risk_factors) if self.primary_risk_factors else "Nominal"
        return (
            f"[RiskReport | Level: {self.risk_level:8s} | Score: {self.overall_risk_score:.3f}] "
            f"Congestion={self.congestion_risk:.2f}, "
            f"Safety={self.safety_hazard_risk:.2f}, "
            f"Emergency={self.emergency_delay_risk:.2f} | "
            f"Factors: [{factors}]"
        )


# ---------------------------------------------------------------------------
# RiskAssessor Engine
# ---------------------------------------------------------------------------

class RiskAssessor:
    """
    Unified Risk Assessment Engine.

    Parameters
    ----------
    risk_model : RiskFNN
    predictor : TrafficPredictor | None
    device : torch.device
    """

    def __init__(
        self,
        risk_model: RiskFNN,
        predictor: Optional[TrafficPredictor] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.risk_model = risk_model.to(self.device)
        self.risk_model.eval()
        self.predictor = predictor
        self.fe = FeatureExtractor()

    @classmethod
    def from_checkpoints(
        cls,
        risk_checkpoint: Union[str, Path] = _ROOT / "models" / "best_risk_model.pt",
        lstm_checkpoint: Optional[Union[str, Path]] = _ROOT / "models" / "best_lstm_model.pt",
        device: str = "cpu",
    ) -> "RiskAssessor":
        """
        Factory method to instantiate a RiskAssessor from saved checkpoint files.
        """
        r_path = Path(risk_checkpoint)
        if not r_path.exists():
            raise FileNotFoundError(f"Risk model checkpoint not found: {r_path}")

        dev = torch.device(device)
        r_ckpt = torch.load(str(r_path), map_location=dev, weights_only=False)

        config = r_ckpt["model_config"]
        risk_model = RiskFNN(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            output_dim=config["output_dim"],
            dropout=config.get("dropout", 0.0),
        )
        risk_model.load_state_dict(r_ckpt["state_dict"])
        risk_model.to(dev)
        risk_model.eval()

        predictor: Optional[TrafficPredictor] = None
        if lstm_checkpoint:
            l_path = Path(lstm_checkpoint)
            if l_path.exists():
                predictor = TrafficPredictor.from_checkpoint(l_path, device=device)

        return cls(risk_model=risk_model, predictor=predictor, device=device)

    # ------------------------------------------------------------------
    # Inference Methods
    # ------------------------------------------------------------------

    def assess_state_and_forecast(
        self,
        current_state: CityState,
        forecast: Dict[str, Sequence[float]],
    ) -> RiskReport:
        """
        Compute risk report directly from current state and a forecast dictionary.
        """
        # 1. Construct 29-dim feature vector
        feat_vec = construct_risk_features(current_state, forecast, self.fe)
        x = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)

        # 2. Neural forward pass
        with torch.no_grad():
            preds = self.risk_model(x).squeeze(0).cpu().numpy()

        r_overall = float(np.clip(preds[0], 0.0, 1.0))
        r_cong    = float(np.clip(preds[1], 0.0, 1.0))
        r_safety  = float(np.clip(preds[2], 0.0, 1.0))
        r_emg     = float(np.clip(preds[3], 0.0, 1.0))

        # 3. Factor identification & recommendations
        factors: List[str] = []
        actions: List[str] = []

        if current_state.accident:
            factors.append("ACTIVE_ACCIDENT_ON_ROADWAY")
            actions.append("Dispatch emergency services & activate rerouting.")
        if current_state.road_blockage:
            factors.append("LANE_OR_ROAD_BLOCKAGE")
            actions.append("Adjust signal splits to clear upstream bottleneck.")
        if current_state.emergency_vehicle_count > 0:
            factors.append(f"EMERGENCY_VEHICLES_ACTIVE (count={current_state.emergency_vehicle_count})")
            actions.append("Trigger emergency vehicle preemption green corridor.")
        if current_state.rain_intensity > 0.5:
            factors.append(f"HEAVY_RAIN (intensity={current_state.rain_intensity:.2f})")
            actions.append("Activate wet-pavement speed advisory.")
        if current_state.fog_intensity > 0.4:
            factors.append(f"DENSE_FOG (intensity={current_state.fog_intensity:.2f})")
            actions.append("Activate low-visibility warning systems.")
        if r_cong > 0.55:
            factors.append("HIGH_CONGESTION_TREND")
            actions.append("Extend arterial green phase duration.")

        if not factors:
            factors.append("NOMINAL_TRAFFIC_CONDITIONS")
            actions.append("Maintain standard adaptive signal schedule.")

        level = classify_risk_level(r_overall)

        # Forecast summary stats
        dens = forecast.get("traffic_density", [current_state.traffic_density])
        spds = forecast.get("average_speed", [current_state.average_speed])
        ques = forecast.get("queue_length", [current_state.queue_length])

        forecast_summary = {
            "peak_predicted_density": float(max(dens)),
            "min_predicted_speed": float(min(spds)),
            "max_predicted_queue": float(max(ques)),
        }

        return RiskReport(
            overall_risk_score=round(r_overall, 4),
            risk_level=level.value,
            congestion_risk=round(r_cong, 4),
            safety_hazard_risk=round(r_safety, 4),
            emergency_delay_risk=round(r_emg, 4),
            primary_risk_factors=factors,
            recommended_actions=actions,
            forecast_summary=forecast_summary,
        )

    def assess(
        self,
        sequence: Sequence[CityState],
    ) -> RiskReport:
        """
        Assess risk from a historical sequence of CityStates (length >= seq_len).
        Runs the temporal LSTM predictor automatically to forecast future states.
        """
        if not sequence:
            raise ValueError("CityState sequence cannot be empty.")

        current_state = sequence[-1]

        if self.predictor is not None:
            # Need matching sequence length
            if len(sequence) >= self.predictor.seq_len:
                seq_in = list(sequence[-self.predictor.seq_len:])
                forecast = self.predictor.predict(seq_in)
            else:
                # If sequence is shorter, repeat initial state to match seq_len
                pad_len = self.predictor.seq_len - len(sequence)
                seq_in = [sequence[0]] * pad_len + list(sequence)
                forecast = self.predictor.predict(seq_in)
        else:
            # Fallback: forecast mirrors current state
            forecast = {
                "traffic_density": [current_state.traffic_density] * 5,
                "average_speed": [current_state.average_speed] * 5,
                "queue_length": [current_state.queue_length] * 5,
            }

        return self.assess_state_and_forecast(current_state, forecast)
