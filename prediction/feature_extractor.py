"""
feature_extractor.py — FeatureExtractor
========================================
Converts a CityState dataclass into a deterministic, normalised float32
numpy vector that the LSTM can consume directly.

Design decisions
----------------
* All features are mapped to [0, 1] using fixed domain bounds from
  mock_city.py so the extractor is completely stateless — no fitting step,
  no risk of train/test leakage through the feature extraction itself.

* Normalisation uses known physical bounds rather than data-driven min-max
  so that a single CityState (seq-len = 1) can be encoded without fitting.

* traffic_signal_state is one-hot encoded over 4 categories:
      [green, yellow, red, flashing]

Feature vector (11 raw inputs → 14 floats after one-hot):
  Index  Feature                    Normalisation
  ─────  ─────────────────────────  ─────────────────────────────────────
  0      vehicle_count              / 500
  1      average_speed              / 120
  2      traffic_density            already [0,1]
  3      queue_length               / 1000
  4      pedestrian_count           / 300
  5      rain_intensity             already [0,1]
  6      fog_intensity              already [0,1]
  7      road_blockage              0 or 1  (bool → float)
  8      accident                   0 or 1
  9      emergency_vehicle_count    / 10
  10-13  traffic_signal_state       one-hot [green, yellow, red, flashing]
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.mock_city import CityState


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Input feature names (in order, matching the vector produced by extract())
FEATURE_NAMES: List[str] = [
    "vehicle_count",
    "average_speed",
    "traffic_density",
    "queue_length",
    "pedestrian_count",
    "rain_intensity",
    "fog_intensity",
    "road_blockage",
    "accident",
    "emergency_vehicle_count",
    "signal_green",
    "signal_yellow",
    "signal_red",
    "signal_flashing",
]

# Names of the three quantities the LSTM is trained to predict
TARGET_NAMES: List[str] = [
    "traffic_density",
    "average_speed",
    "queue_length",
]

# Fixed domain bounds for normalisation (matching CityState constraints)
_BOUNDS = {
    "vehicle_count":           500.0,
    "average_speed":           120.0,
    "queue_length":            1000.0,
    "pedestrian_count":        300.0,
    "emergency_vehicle_count": 10.0,
}

_SIGNAL_ORDER = ["green", "yellow", "red", "flashing"]

# Input dimensionality (= len(FEATURE_NAMES))
FEATURE_DIM: int = len(FEATURE_NAMES)

# Target dimensionality (= len(TARGET_NAMES))
TARGET_DIM: int = len(TARGET_NAMES)


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Stateless converter: CityState  →  float32 numpy vector of shape (14,).

    Usage
    -----
    >>> fe = FeatureExtractor()
    >>> vec = fe.extract(state)            # shape (14,)
    >>> mat = fe.extract_sequence(states)  # shape (T, 14)

    Inverse (targets only)
    ----------------------
    >>> targets = fe.extract_targets(state)   # shape (3,)  — unnormalised
    >>> reconstructed = fe.inverse_targets(targets)  # dict with real units
    """

    # ------------------------------------------------------------------
    # Forward: CityState → feature vector
    # ------------------------------------------------------------------

    def extract(self, state: CityState) -> np.ndarray:
        """
        Convert a single CityState to a normalised float32 vector.

        Parameters
        ----------
        state : CityState

        Returns
        -------
        np.ndarray, shape (FEATURE_DIM,), dtype float32
        """
        # One-hot signal
        signal_oh = [0.0, 0.0, 0.0, 0.0]
        sig = state.traffic_signal_state.lower()
        if sig in _SIGNAL_ORDER:
            signal_oh[_SIGNAL_ORDER.index(sig)] = 1.0

        vec = np.array([
            state.vehicle_count           / _BOUNDS["vehicle_count"],
            state.average_speed           / _BOUNDS["average_speed"],
            float(state.traffic_density),
            state.queue_length            / _BOUNDS["queue_length"],
            state.pedestrian_count        / _BOUNDS["pedestrian_count"],
            float(state.rain_intensity),
            float(state.fog_intensity),
            float(state.road_blockage),
            float(state.accident),
            state.emergency_vehicle_count / _BOUNDS["emergency_vehicle_count"],
            *signal_oh,
        ], dtype=np.float32)

        assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {vec.shape}"
        return vec

    def extract_sequence(self, states: Sequence[CityState]) -> np.ndarray:
        """
        Convert a list/sequence of CityStates to a float32 matrix.

        Parameters
        ----------
        states : sequence of CityState, length T

        Returns
        -------
        np.ndarray, shape (T, FEATURE_DIM), dtype float32
        """
        return np.stack([self.extract(s) for s in states], axis=0)

    # ------------------------------------------------------------------
    # Target extraction (the three values the LSTM predicts)
    # ------------------------------------------------------------------

    def extract_targets(self, state: CityState) -> np.ndarray:
        """
        Extract the three prediction targets from a CityState.

        Returns normalised values in [0, 1] matching model output scale.

        Returns
        -------
        np.ndarray, shape (TARGET_DIM,), dtype float32
            [traffic_density, average_speed_normalised, queue_length_normalised]
        """
        return np.array([
            float(state.traffic_density),
            state.average_speed / _BOUNDS["average_speed"],
            state.queue_length  / _BOUNDS["queue_length"],
        ], dtype=np.float32)

    def extract_target_sequence(self, states: Sequence[CityState]) -> np.ndarray:
        """
        Extract targets for a sequence of states.

        Returns
        -------
        np.ndarray, shape (T, TARGET_DIM), dtype float32
        """
        return np.stack([self.extract_targets(s) for s in states], axis=0)

    # ------------------------------------------------------------------
    # Inverse: normalised target vector → human-readable dict
    # ------------------------------------------------------------------

    def inverse_targets(self, targets: np.ndarray) -> dict:
        """
        Convert a normalised target vector (or matrix) back to real units.

        Parameters
        ----------
        targets : np.ndarray, shape (TARGET_DIM,) or (T, TARGET_DIM)

        Returns
        -------
        dict with keys: traffic_density, average_speed, queue_length
            Values are float (1-D) or list[float] (2-D input).
        """
        def _to_real(row: np.ndarray) -> dict:
            return {
                "traffic_density": float(np.clip(row[0], 0.0, 1.0)),
                "average_speed":   float(np.clip(row[1], 0.0, 1.0) * _BOUNDS["average_speed"]),
                "queue_length":    float(np.clip(row[2], 0.0, 1.0) * _BOUNDS["queue_length"]),
            }

        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim == 1:
            return _to_real(targets)

        # (T, 3) → list of dicts, then re-structure as dict of lists
        rows = [_to_real(targets[i]) for i in range(targets.shape[0])]
        return {k: [r[k] for r in rows] for k in TARGET_NAMES}
