"""
risk/feature_builder.py — Risk Feature Builder & Target Generation
==================================================================
Fuses current CityState observations with LSTM future traffic forecasts
into a deterministic numerical feature vector, and computes the synthetic
urban disruption severity training targets.

Features Produced (23 Dimensions)
---------------------------------
  --- Current State (14 features) ---
  0: vehicle_count / 500.0
  1: average_speed / 120.0
  2: traffic_density [0, 1]
  3: queue_length / 1000.0
  4: pedestrian_count / 300.0
  5: rain_intensity [0, 1]
  6: fog_intensity [0, 1]
  7: road_blockage (0 or 1)
  8: accident (0 or 1)
  9: emergency_vehicle_count / 10.0
  10: signal_green (0 or 1)
  11: signal_yellow (0 or 1)
  12: signal_red (0 or 1)
  13: signal_flashing (0 or 1)

  --- Forecast Summary (6 features) ---
  14: predicted_max_traffic_density [0, 1]
  15: predicted_mean_traffic_density [0, 1]
  16: predicted_min_average_speed / 120.0
  17: predicted_mean_average_speed / 120.0
  18: predicted_max_queue_length / 1000.0
  19: predicted_mean_queue_length / 1000.0

  --- Dynamic Changes & Trends (3 features) ---
  20: traffic_density_change (pred_mean_density - curr_density)
  21: speed_change ((pred_mean_speed - curr_speed) / 120.0)
  22: queue_growth ((pred_mean_queue - curr_queue) / 1000.0)

Risk Severity Levels (Target: 0–100)
------------------------------------
  0–24   : LOW
  25–49  : MEDIUM
  50–74  : HIGH
  75–100 : CRITICAL
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.mock_city import CityState
from prediction.feature_extractor import FeatureExtractor, FEATURE_DIM


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


RISK_FEATURE_NAMES: List[str] = [
    # Current state features
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
    # Forecast summary features
    "predicted_max_traffic_density",
    "predicted_mean_traffic_density",
    "predicted_min_average_speed",
    "predicted_mean_average_speed",
    "predicted_max_queue_length",
    "predicted_mean_queue_length",
    # Dynamic trend features
    "traffic_density_change",
    "speed_change",
    "queue_growth",
]

RISK_FEATURE_DIM: int = len(RISK_FEATURE_NAMES)  # 23


def classify_risk_level(score: float) -> str:
    """
    Map a continuous risk score in [0, 100] to a categorical risk level.

    Thresholds
    ----------
    0 - 24.99  : LOW
    25 - 49.99 : MEDIUM
    50 - 74.99 : HIGH
    75 - 100.0 : CRITICAL
    """
    score = float(np.clip(score, 0.0, 100.0))
    if score < 25.0:
        return RiskLevel.LOW.value
    elif score < 50.0:
        return RiskLevel.MEDIUM.value
    elif score < 75.0:
        return RiskLevel.HIGH.value
    else:
        return RiskLevel.CRITICAL.value


# ---------------------------------------------------------------------------
# Target Generation Function
# ---------------------------------------------------------------------------

def generate_synthetic_risk_target(
    current_state: CityState,
    future_prediction: Dict[str, Sequence[float]],
    noise_std: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Construct a synthetic 'urban disruption severity' target score in [0, 100].

    NOTE: This is a prototype simulation target designed for CityTwin AI Unit I.
    It represents urban disruption severity synthesized from physics-grounded
    heuristics across congestion, forecast trends, adverse weather, and incidents.
    It does NOT claim to represent empirical real-world crash probabilities.

    Formulation:
      - Current Congestion: density, queue length, speed deficit (up to 25 pts)
      - Predicted Congestion & Trends: max forecast density, queue growth (up to 20 pts)
      - Incidents: accident (+30 pts), road blockage (+10 pts)
      - Weather: rain intensity (+8 pts), fog intensity (+7 pts)
      - Emergency Response: EV count & bottlenecks (up to 15 pts)
      - Pedestrian Conflict: ped count * speed (up to 5 pts)
      - Controlled Gaussian noise added for machine learning training.

    Parameters
    ----------
    current_state : CityState
    future_prediction : dict with 'traffic_density', 'average_speed', 'queue_length'
    noise_std : float (default 0.0)
    rng : np.random.RandomState | None

    Returns
    -------
    float in [0.0, 100.0]
    """
    # 1. Current physical metrics
    curr_density = float(np.clip(current_state.traffic_density, 0.0, 1.0))
    curr_queue_norm = float(np.clip(current_state.queue_length / 1000.0, 0.0, 1.0))
    curr_speed_deficit = float(np.clip(max(0.0, (60.0 - current_state.average_speed) / 60.0), 0.0, 1.0))
    rain = float(np.clip(current_state.rain_intensity, 0.0, 1.0))
    fog = float(np.clip(current_state.fog_intensity, 0.0, 1.0))
    has_accident = 1.0 if current_state.accident else 0.0
    has_blockage = 1.0 if current_state.road_blockage else 0.0
    ev_count = current_state.emergency_vehicle_count
    ped_norm = float(np.clip(current_state.pedestrian_count / 300.0, 0.0, 1.0))

    # 2. Predicted future metrics
    pred_densities = future_prediction.get("traffic_density", [curr_density])
    pred_speeds = future_prediction.get("average_speed", [current_state.average_speed])
    pred_queues = future_prediction.get("queue_length", [current_state.queue_length])

    pred_max_density = float(np.clip(max(pred_densities), 0.0, 1.0))
    pred_mean_queue = float(np.mean(pred_queues))
    queue_growth = float(max(0.0, (pred_mean_queue - current_state.queue_length) / 1000.0))
    pred_min_speed_deficit = float(np.clip(max(0.0, (60.0 - min(pred_speeds)) / 60.0), 0.0, 1.0))

    # 3. Weighted component summation (Max ~ 100)
    congestion_score = 10.0 * curr_density + 8.0 * curr_queue_norm + 7.0 * curr_speed_deficit
    forecast_trend_score = 10.0 * pred_max_density + 5.0 * queue_growth + 5.0 * pred_min_speed_deficit
    incident_score = 30.0 * has_accident + 10.0 * has_blockage
    weather_score = 8.0 * rain + 7.0 * fog
    emergency_score = min(15.0, ev_count * 3.5 * (0.5 + 0.5 * curr_density + 0.3 * has_blockage))
    ped_conflict_score = 5.0 * ped_norm * (current_state.average_speed / 80.0)

    raw_score = (
        congestion_score
        + forecast_trend_score
        + incident_score
        + weather_score
        + emergency_score
        + ped_conflict_score
    )

    # Accident baseline floor
    if has_accident:
        raw_score = max(raw_score, 75.0)
    elif has_blockage and ev_count >= 2:
        raw_score = max(raw_score, 65.0)

    # Controlled noise
    if noise_std > 0.0:
        generator = rng or np.random
        noise = float(generator.normal(0.0, noise_std))
        raw_score += noise

    return float(np.clip(raw_score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Feature Builder Class
# ---------------------------------------------------------------------------

class RiskFeatureBuilder:
    """
    Deterministic feature builder combining current CityState observables
    and LSTM future traffic predictions into a 23-dimensional feature vector.
    """

    def __init__(self) -> None:
        self._fe = FeatureExtractor()

    def build_features(
        self,
        current_state: CityState,
        future_prediction: Dict[str, Sequence[float]],
    ) -> np.ndarray:
        """
        Build the 23-dimensional risk feature vector.

        Parameters
        ----------
        current_state : CityState
        future_prediction : dict with 'traffic_density', 'average_speed', 'queue_length'

        Returns
        -------
        np.ndarray of shape (23,), dtype float32
        """
        curr_vec = self._fe.extract(current_state)  # (14,)

        pred_densities = np.array(
            future_prediction.get("traffic_density", [current_state.traffic_density]),
            dtype=np.float32,
        )
        pred_speeds = np.array(
            future_prediction.get("average_speed", [current_state.average_speed]),
            dtype=np.float32,
        )
        pred_queues = np.array(
            future_prediction.get("queue_length", [current_state.queue_length]),
            dtype=np.float32,
        )

        pred_max_density = float(np.max(pred_densities))
        pred_mean_density = float(np.mean(pred_densities))
        pred_min_speed_norm = float(np.min(pred_speeds) / 120.0)
        pred_mean_speed_norm = float(np.mean(pred_speeds) / 120.0)
        pred_max_queue_norm = float(np.max(pred_queues) / 1000.0)
        pred_mean_queue_norm = float(np.mean(pred_queues) / 1000.0)

        summary_features = [
            np.clip(pred_max_density, 0.0, 1.0),
            np.clip(pred_mean_density, 0.0, 1.0),
            np.clip(pred_min_speed_norm, 0.0, 1.0),
            np.clip(pred_mean_speed_norm, 0.0, 1.0),
            np.clip(pred_max_queue_norm, 0.0, 1.0),
            np.clip(pred_mean_queue_norm, 0.0, 1.0),
        ]

        curr_density = float(current_state.traffic_density)
        curr_speed = float(current_state.average_speed)
        curr_queue = float(current_state.queue_length)

        density_change = pred_mean_density - curr_density
        speed_change = (float(np.mean(pred_speeds)) - curr_speed) / 120.0
        queue_growth = (float(np.mean(pred_queues)) - curr_queue) / 1000.0

        change_features = [
            float(density_change),
            float(speed_change),
            float(queue_growth),
        ]

        full_vec = np.concatenate(
            [curr_vec, np.array(summary_features, dtype=np.float32), np.array(change_features, dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

        assert full_vec.shape == (RISK_FEATURE_DIM,), f"Expected ({RISK_FEATURE_DIM},), got {full_vec.shape}"
        return full_vec
