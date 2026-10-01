"""
rl/state_encoder.py — Compact State Representation for RL Traffic Control
==========================================================================
Encodes the current CityState physical observables, the 5-step LSTM forecast,
the FNN urban disruption risk score, and temporal trends into a compact,
normalized 20-dimensional float32 state vector.

State Dimensions (20 Features)
------------------------------
  --- Current Observables (10 features) ---
  0: traffic_density [0, 1]
  1: average_speed / 120.0
  2: queue_length / 1000.0
  3: vehicle_count / 500.0
  4: pedestrian_count / 300.0
  5: rain_intensity [0, 1]
  6: fog_intensity [0, 1]
  7: road_blockage (0 or 1)
  8: accident (0 or 1)
  9: emergency_vehicle_count / 10.0

  --- LSTM Forecast Summaries (6 features) ---
  10: predicted_mean_traffic_density [0, 1]
  11: predicted_max_traffic_density [0, 1]
  12: predicted_mean_speed / 120.0
  13: predicted_min_speed / 120.0
  14: predicted_mean_queue / 1000.0
  15: predicted_max_queue / 1000.0

  --- FNN Disruption Risk (1 feature) ---
  16: risk_score / 100.0

  --- Dynamic Trends (3 features) ---
  17: traffic_density_change (pred_mean_density - curr_density)
  18: speed_change ((pred_mean_speed - curr_speed) / 120.0)
  19: queue_growth ((pred_mean_queue - curr_queue) / 1000.0)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.mock_city import CityState


STATE_FEATURE_NAMES: List[str] = [
    # Current
    "traffic_density",
    "average_speed",
    "queue_length",
    "vehicle_count",
    "pedestrian_count",
    "rain_intensity",
    "fog_intensity",
    "road_blockage",
    "accident",
    "emergency_vehicle_count",
    # Forecast
    "predicted_mean_traffic_density",
    "predicted_max_traffic_density",
    "predicted_mean_speed",
    "predicted_min_speed",
    "predicted_mean_queue",
    "predicted_max_queue",
    # Risk
    "risk_score",
    # Trends
    "traffic_density_change",
    "speed_change",
    "queue_growth",
]

STATE_DIM: int = len(STATE_FEATURE_NAMES)  # 20


class StateEncoder:
    """
    Deterministic state encoder converting physical city state,
    LSTM forecast, and FNN disruption risk into a 20-dim RL state vector.
    """

    def encode(
        self,
        current_state: CityState,
        forecast: Dict[str, Sequence[float]],
        risk_score: float,
    ) -> np.ndarray:
        """
        Encode all inputs into a compact 20-dimensional float32 vector.

        Parameters
        ----------
        current_state : CityState
            Current city observation snapshot.
        forecast : dict
            LSTM future predictions (traffic_density, average_speed, queue_length).
        risk_score : float
            FNN disruption risk score in [0, 100].

        Returns
        -------
        np.ndarray of shape (20,), dtype float32
        """
        # 1. Current observables
        c_density = float(np.clip(current_state.traffic_density, 0.0, 1.0))
        c_speed = float(np.clip(current_state.average_speed / 120.0, 0.0, 1.0))
        c_queue = float(np.clip(current_state.queue_length / 1000.0, 0.0, 1.0))
        c_vehicles = float(np.clip(current_state.vehicle_count / 500.0, 0.0, 1.0))
        c_peds = float(np.clip(current_state.pedestrian_count / 300.0, 0.0, 1.0))
        c_rain = float(np.clip(current_state.rain_intensity, 0.0, 1.0))
        c_fog = float(np.clip(current_state.fog_intensity, 0.0, 1.0))
        c_block = 1.0 if current_state.road_blockage else 0.0
        c_acc = 1.0 if current_state.accident else 0.0
        c_ev = float(np.clip(current_state.emergency_vehicle_count / 10.0, 0.0, 1.0))

        current_features = [
            c_density,
            c_speed,
            c_queue,
            c_vehicles,
            c_peds,
            c_rain,
            c_fog,
            c_block,
            c_acc,
            c_ev,
        ]

        # 2. Forecast summaries
        f_densities = forecast.get("traffic_density", [current_state.traffic_density])
        f_speeds = forecast.get("average_speed", [current_state.average_speed])
        f_queues = forecast.get("queue_length", [current_state.queue_length])

        f_mean_density = float(np.clip(np.mean(f_densities), 0.0, 1.0))
        f_max_density = float(np.clip(np.max(f_densities), 0.0, 1.0))

        f_mean_speed = float(np.clip(np.mean(f_speeds) / 120.0, 0.0, 1.0))
        f_min_speed = float(np.clip(np.min(f_speeds) / 120.0, 0.0, 1.0))

        f_mean_queue = float(np.clip(np.mean(f_queues) / 1000.0, 0.0, 1.0))
        f_max_queue = float(np.clip(np.max(f_queues) / 1000.0, 0.0, 1.0))

        forecast_features = [
            f_mean_density,
            f_max_density,
            f_mean_speed,
            f_min_speed,
            f_mean_queue,
            f_max_queue,
        ]

        # 3. Risk score normalized to [0, 1]
        r_norm = float(np.clip(risk_score / 100.0, 0.0, 1.0))

        # 4. Trends
        d_change = float(f_mean_density - c_density)
        s_change = float(f_mean_speed - c_speed)
        q_growth = float(f_mean_queue - c_queue)

        trend_features = [d_change, s_change, q_growth]

        vec = np.array(
            current_features + forecast_features + [r_norm] + trend_features,
            dtype=np.float32,
        )

        assert vec.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {vec.shape}"
        return vec
