"""
predictor.py — TrafficPredictor
=================================
High-level inference API for the trained LSTM.

Callers supply a sequence of CityState objects (or a pre-built numpy
feature matrix) and receive a structured prediction dict — no knowledge
of PyTorch, tensors, or normalisation required.

CityTwin pipeline context
--------------------------
  [Observation layer — from Stage 3 CNN or mock simulation]
          ↓
  List[CityState]   (length = seq_len, e.g. last 10 ticks)
          ↓
  TrafficPredictor.predict()
          ↓
  {
    "traffic_density": [0.42, 0.45, 0.47, 0.50, 0.52],  ← horizon steps
    "average_speed":   [38.1, 35.7, 33.2, 30.8, 28.5],  km/h
    "queue_length":    [214.0, 242.0, 271.0, 300.0, 325.0],  metres
  }
          ↓
  [Risk Assessment Module — Stage 4]
          ↓
  [RL Traffic Controller — Stage 5]

Usage
-----
>>> from prediction import TrafficPredictor
>>> predictor = TrafficPredictor.from_checkpoint("models/best_lstm_model.pt")
>>> predictions = predictor.predict(recent_states)   # List[CityState]
>>> print(predictions["average_speed"])
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.mock_city import CityState
from prediction.feature_extractor import FeatureExtractor, FEATURE_DIM, TARGET_NAMES
from prediction.lstm_model import TrafficLSTM

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TrafficPredictor:
    """
    Load a trained TrafficLSTM checkpoint and serve predictions.

    Parameters
    ----------
    model : TrafficLSTM
        Loaded model (already in eval mode).
    seq_len : int
        Expected input sequence length.
    device : torch.device
        Device to run inference on.

    Instantiation
    -------------
    Use the class method ``from_checkpoint()`` rather than the constructor.
    """

    def __init__(
        self,
        model:    "TrafficLSTM",
        seq_len:  int,
        device:   "torch.device",
    ) -> None:
        self._model = model
        self._seq_len = seq_len
        self._device = device
        self._fe = FeatureExtractor()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
    ) -> "TrafficPredictor":
        """
        Load a saved checkpoint and return a ready-to-use predictor.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the .pt file saved by train_lstm.py.
        device : str
            Torch device string, e.g. "cpu" or "cuda".

        Returns
        -------
        TrafficPredictor
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install it with: pip install torch")

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        dev = torch.device(device)
        ckpt = torch.load(str(path), map_location=dev, weights_only=False)

        config = ckpt["model_config"]
        model = TrafficLSTM(
            input_size  = config["input_size"],
            hidden_size = config["hidden_size"],
            num_layers  = config["num_layers"],
            horizon     = config["horizon"],
            target_dim  = config["target_dim"],
            dropout     = config.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev)
        model.eval()

        seq_len = ckpt.get("seq_len", 10)
        return cls(model=model, seq_len=seq_len, device=dev)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        sequence: Union[Sequence[CityState], np.ndarray],
    ) -> Dict[str, List[float]]:
        """
        Predict future traffic conditions from a historical sequence.

        Parameters
        ----------
        sequence : list of CityState or np.ndarray of shape (seq_len, FEATURE_DIM)
            The most recent ``seq_len`` observations.

        Returns
        -------
        dict with keys: traffic_density, average_speed, queue_length
            Each value is a list of ``horizon`` floats in real-world units:
              traffic_density — [0, 1] (normalised)
              average_speed   — km/h
              queue_length    — metres
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        # --- Convert input to numpy feature matrix ---
        if isinstance(sequence, np.ndarray):
            feat = sequence.astype(np.float32)
        else:
            # List[CityState]
            feat = self._fe.extract_sequence(list(sequence))

        # --- Validate shape ---
        if feat.ndim != 2 or feat.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature matrix of shape (seq_len, {FEATURE_DIM}), "
                f"got {feat.shape}"
            )
        if feat.shape[0] != self._seq_len:
            raise ValueError(
                f"Predictor expects seq_len={self._seq_len} steps, "
                f"got {feat.shape[0]}. "
                "Either provide exactly seq_len states or instantiate a new "
                "predictor with a matching checkpoint."
            )

        # --- Run inference ---
        x = torch.from_numpy(feat).unsqueeze(0).to(self._device)  # (1, T, F)
        with torch.no_grad():
            pred = self._model(x)   # (1, horizon, target_dim)

        pred_np = pred.squeeze(0).cpu().numpy()  # (horizon, target_dim)

        # --- Validate finiteness ---
        if not np.all(np.isfinite(pred_np)):
            raise RuntimeError(
                "LSTM produced non-finite predictions. "
                "Check model weights and input scaling."
            )

        # --- Convert to real units ---
        result = self._fe.inverse_targets(pred_np)  # dict of lists
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Expected number of input timesteps."""
        return self._seq_len

    @property
    def horizon(self) -> int:
        """Number of future timesteps predicted."""
        return self._model.horizon

    @property
    def model(self) -> "TrafficLSTM":
        """Underlying TrafficLSTM (read-only access for inspection)."""
        return self._model

    def __repr__(self) -> str:
        return (
            f"TrafficPredictor("
            f"seq_len={self._seq_len}, "
            f"horizon={self._model.horizon}, "
            f"device={self._device})"
        )
