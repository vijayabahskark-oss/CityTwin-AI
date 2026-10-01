"""
risk/predictor.py — High-Level Risk Predictor
==============================================
Provides a clean, standalone inference interface for the trained RiskMLP model.

Callers provide the current `CityState` and the LSTM `future_prediction` dict
and receive structured risk outputs without needing PyTorch knowledge.

Usage
-----
>>> from risk.predictor import RiskPredictor
>>> predictor = RiskPredictor.from_checkpoint("models/risk_model.pt")
>>> result = predictor.predict(current_state, future_prediction)
>>> print(result)
{
    "risk_score": 38.45,
    "risk_level": "MEDIUM"
}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.mock_city import CityState
from risk.feature_builder import (
    RiskFeatureBuilder,
    classify_risk_level,
    RISK_FEATURE_DIM,
)
from risk.risk_model import RiskMLP


class RiskPredictor:
    """
    Inference interface wrapping the trained RiskMLP model.

    Parameters
    ----------
    model : RiskMLP
        Loaded PyTorch model in eval mode.
    device : torch.device
        Device used for inference.
    """

    def __init__(
        self,
        model: RiskMLP,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._builder = RiskFeatureBuilder()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = _ROOT / "models" / "risk_model.pt",
        device: str = "cpu",
    ) -> "RiskPredictor":
        """
        Instantiate RiskPredictor from a saved model checkpoint file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the saved .pt checkpoint.
        device : str
            Device string ("cpu" or "cuda").

        Returns
        -------
        RiskPredictor instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Risk model checkpoint not found: {path}")

        dev = torch.device(device)
        ckpt = torch.load(str(path), map_location=dev, weights_only=False)

        config = ckpt.get("model_config", {})
        model = RiskMLP(
            input_dim=config.get("input_dim", RISK_FEATURE_DIM),
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev)
        model.eval()

        return cls(model=model, device=dev)

    def predict(
        self,
        current_state: CityState,
        future_prediction: Dict[str, Sequence[float]],
    ) -> Dict[str, Union[float, str]]:
        """
        Predict urban disruption risk score (0-100) and risk level category.

        Parameters
        ----------
        current_state : CityState
            Current city observation snapshot.
        future_prediction : dict
            Future trajectory predictions from LSTM predictor containing:
            - 'traffic_density': sequence of floats
            - 'average_speed': sequence of floats (km/h)
            - 'queue_length': sequence of floats (metres)

        Returns
        -------
        dict with keys:
            'risk_score' : float in [0.0, 100.0]
            'risk_level' : str ("LOW" | "MEDIUM" | "HIGH" | "CRITICAL")
        """
        # 1. Build 23-dim feature vector
        feat_vec = self._builder.build_features(current_state, future_prediction)

        # 2. Convert to tensor
        x = torch.from_numpy(feat_vec).unsqueeze(0).to(self._device)

        # 3. Model forward pass
        with torch.no_grad():
            norm_score = self._model(x).item()  # [0, 1]

        # 4. Scale to [0, 100] and classify
        risk_score = float(np.clip(norm_score * 100.0, 0.0, 100.0))
        risk_level = classify_risk_level(risk_score)

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
        }

    @property
    def model(self) -> RiskMLP:
        """Access underlying PyTorch model."""
        return self._model

    def __repr__(self) -> str:
        return f"RiskPredictor(device={self._device}, input_dim={self._model.input_dim})"
