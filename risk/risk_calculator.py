"""
risk_calculator.py — Ground Truth Risk Formulation & Heuristics
================================================================
Defines domain-grounded mathematical formulations to compute multi-dimensional
urban risk from current physical observations and future forecasted states.

These formulations serve two roles:
  1. Generating ground-truth training targets for the RiskFNN model.
  2. Providing deterministic rule-based verification and advisory logic.

Risk Dimensions
---------------
  1. Congestion Risk (R_cong):
     Driven by current density, queue length, speed deficit, and predicted
     peak density/queue increases over the next horizon.

  2. Safety Hazard Risk (R_safety):
     Driven by active accidents, road blockages, heavy precipitation, dense fog,
     and interactions between pedestrian density and high vehicle speeds.

  3. Emergency Delay Risk (R_emg):
     Driven by presence of emergency vehicles, gridlock severity, and blockages
     hindering emergency response routes.

  4. Overall Risk Index (R_overall):
     Synthesized composite score combining all sub-risks, with emergency/accident
     priority escalations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.mock_city import CityState


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


RISK_TARGET_NAMES: List[str] = [
    "overall_risk",
    "congestion_risk",
    "safety_risk",
    "emergency_risk",
]

RISK_TARGET_DIM: int = len(RISK_TARGET_NAMES)


def classify_risk_level(score: float) -> RiskLevel:
    """Categorize continuous score in [0, 1] into discrete RiskLevel."""
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.30:
        return RiskLevel.LOW
    elif score < 0.60:
        return RiskLevel.MODERATE
    elif score < 0.85:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Ground Truth Computation
# ---------------------------------------------------------------------------

def compute_ground_truth_risk(
    current_state: CityState,
    forecast: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, Union[float, str, List[str]]]:
    """
    Compute ground truth multi-dimensional risk scores from a CityState
    and optional future forecasted trajectories.

    Parameters
    ----------
    current_state : CityState
        Current observation snapshot.
    forecast : dict | None
        Predicted future trajectories with keys:
          'traffic_density' (list of floats in [0, 1])
          'average_speed' (list of floats in km/h)
          'queue_length' (list of floats in metres)
        If None, computation defaults to current state observations only.

    Returns
    -------
    dict with keys:
        'overall_risk' : float in [0, 1]
        'congestion_risk' : float in [0, 1]
        'safety_risk' : float in [0, 1]
        'emergency_risk' : float in [0, 1]
        'risk_level' : str ("LOW" | "MODERATE" | "HIGH" | "CRITICAL")
        'primary_risk_factors' : list[str]
        'recommended_actions' : list[str]
    """
    # 1. Extract Current Observables
    curr_density = float(np.clip(current_state.traffic_density, 0.0, 1.0))
    curr_queue_norm = float(np.clip(current_state.queue_length / 1000.0, 0.0, 1.0))
    curr_speed_deficit = float(np.clip(max(0.0, (60.0 - current_state.average_speed) / 60.0), 0.0, 1.0))
    ped_norm = float(np.clip(current_state.pedestrian_count / 300.0, 0.0, 1.0))
    rain = float(np.clip(current_state.rain_intensity, 0.0, 1.0))
    fog = float(np.clip(current_state.fog_intensity, 0.0, 1.0))
    has_accident = 1.0 if current_state.accident else 0.0
    has_blockage = 1.0 if current_state.road_blockage else 0.0
    ev_count = current_state.emergency_vehicle_count

    # 2. Extract Forecasted Dynamics (if available)
    if forecast and len(forecast.get("traffic_density", [])) > 0:
        fut_densities = forecast["traffic_density"]
        fut_speeds = forecast["average_speed"]
        fut_queues = forecast["queue_length"]

        fut_peak_density = float(np.clip(max(fut_densities), 0.0, 1.0))
        fut_min_speed = float(np.clip(min(fut_speeds), 0.0, 120.0))
        fut_speed_deficit = float(np.clip(max(0.0, (60.0 - fut_min_speed) / 60.0), 0.0, 1.0))
        fut_peak_queue_norm = float(np.clip(max(fut_queues) / 1000.0, 0.0, 1.0))
    else:
        fut_peak_density = curr_density
        fut_speed_deficit = curr_speed_deficit
        fut_peak_queue_norm = curr_queue_norm

    # 3. Calculate Sub-Risks
    # A. Congestion Risk
    # Fuses current density & queue with predicted congestion buildup
    r_cong = (
        0.25 * curr_density
        + 0.20 * curr_queue_norm
        + 0.15 * curr_speed_deficit
        + 0.25 * fut_peak_density
        + 0.15 * fut_peak_queue_norm
    )
    if has_blockage:
        r_cong = min(1.0, r_cong + 0.15)
    r_cong = float(np.clip(r_cong, 0.0, 1.0))

    # B. Safety Hazard Risk
    # Driven by incidents, severe adverse weather, and pedestrian conflicts
    weather_hazard = 0.60 * rain + 0.40 * fog
    ped_conflict = ped_norm * (current_state.average_speed / 100.0)

    r_safety = (
        0.40 * has_accident
        + 0.20 * has_blockage
        + 0.25 * weather_hazard
        + 0.15 * ped_conflict
    )
    r_safety = float(np.clip(r_safety, 0.0, 1.0))

    # C. Emergency Delay Risk
    # Non-zero when emergency vehicles are present, magnified by congestion/blockages
    if ev_count > 0:
        ev_urgency = min(1.0, ev_count / 3.0)
        impediment = 0.40 * curr_density + 0.35 * curr_queue_norm + 0.25 * has_blockage
        r_emg = 0.30 * ev_urgency + 0.70 * (ev_urgency * impediment)
    else:
        r_emg = 0.0
    r_emg = float(np.clip(r_emg, 0.0, 1.0))

    # 4. Synthesize Overall Composite Risk Score
    # Standard weighted combination
    r_overall = 0.40 * r_cong + 0.35 * r_safety + 0.25 * r_emg

    # Critical incident escalation logic
    if has_accident:
        # Accidents escalate risk to minimum HIGH (>= 0.70)
        r_overall = max(r_overall, 0.70 + 0.25 * r_cong)
    if ev_count >= 2 and (has_blockage or curr_density > 0.7):
        # Impeded active emergency escalates risk to minimum HIGH/CRITICAL (>= 0.75)
        r_overall = max(r_overall, 0.75 + 0.20 * r_emg)
    if rain > 0.6 and fog > 0.5:
        # Severe multi-hazard weather escalation
        r_overall = max(r_overall, 0.65 + 0.25 * r_safety)

    r_overall = float(np.clip(r_overall, 0.0, 1.0))

    # 5. Diagnostic Factors and Actionable Advisories
    factors: List[str] = []
    actions: List[str] = []

    if has_accident:
        factors.append("ACTIVE_ACCIDENT_ON_ROADWAY")
        actions.append("Dispatch emergency services & reroute inbound traffic.")
    if has_blockage:
        factors.append("LANE_OR_ROAD_BLOCKAGE")
        actions.append("Adjust signal splits to clear upstream bottlenecks.")
    if ev_count > 0:
        factors.append(f"ACTIVE_EMERGENCY_VEHICLES (count={ev_count})")
        actions.append("Trigger emergency vehicle preemption green corridor.")
    if rain > 0.5:
        factors.append(f"HEAVY_RAIN (intensity={rain:.2f})")
        actions.append("Activate wet-pavement speed advisory.")
    if fog > 0.4:
        factors.append(f"DENSE_FOG (intensity={fog:.2f})")
        actions.append("Activate low-visibility warning systems.")
    if r_cong > 0.6:
        factors.append("SEVERE_CONGESTION_TREND")
        actions.append("Extend main corridor green time to dissipate queue.")

    if not factors:
        factors.append("NOMINAL_TRAFFIC_CONDITIONS")
        actions.append("Maintain standard adaptive signal scheduling.")

    level = classify_risk_level(r_overall)

    return {
        "overall_risk": r_overall,
        "congestion_risk": r_cong,
        "safety_risk": r_safety,
        "emergency_risk": r_emg,
        "risk_level": level.value,
        "primary_risk_factors": factors,
        "recommended_actions": actions,
    }
