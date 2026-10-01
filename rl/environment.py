"""
rl/environment.py — CityTrafficEnv (Gymnasium-Compatible RL Environment)
========================================================================
Implements a lightweight, discrete-action urban traffic control environment
that chains:
  MockCitySimulator ──► TrafficPredictor (LSTM) ──► RiskPredictor (FNN) ──► StateEncoder

Action Space (6 Discrete Actions)
---------------------------------
  0 = NO_ACTION          : Baseline standard signal control.
  1 = EXTEND_GREEN       : Lengthens green signal to accelerate corridor discharge.
  2 = REDUCE_GREEN       : Shortens arterial green phase to service cross-streets.
  3 = TRAFFIC_DIVERSION  : Diverts upstream volume away from monitored corridor.
  4 = EMERGENCY_PRIORITY : Preempts signals to grant green corridor for emergency vehicles.
  5 = COMBINED_RESPONSE  : Simultaneous traffic diversion + emergency priority preemption.

Reward Function
---------------
Multi-objective reward encouraging high throughput, short queues, rapid emergency
clearance, and penalizing unnecessary interventions:

  R = w_density * (1.0 - density)
    + w_speed   * min(1.0, speed / 60.0)
    + w_queue   * (1.0 - min(1.0, queue / 500.0))
    + R_emergency
    - Cost_action

Where:
  w_density = 1.0, w_speed = 0.8, w_queue = 0.8
  Cost_action: NO_ACTION=0.0, EXTEND/REDUCE=0.05, DIVERSION=0.15, EMG=0.10, COMBINED=0.25
"""

from __future__ import annotations

import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Gymnasium compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    class gym:  # type: ignore
        class Env:
            pass
    class spaces:  # type: ignore
        class Discrete:
            def __init__(self, n: int):
                self.n = n
        class Box:
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from simulation import MockCitySimulator, ScenarioGenerator, ScenarioType
from simulation.mock_city import CityState
from prediction.predictor import TrafficPredictor
from risk.predictor import RiskPredictor
from rl.state_encoder import StateEncoder, STATE_DIM


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

class TrafficAction(IntEnum):
    NO_ACTION = 0
    EXTEND_GREEN = 1
    REDUCE_GREEN = 2
    TRAFFIC_DIVERSION = 3
    EMERGENCY_PRIORITY = 4
    COMBINED_RESPONSE = 5


ACTION_DESCRIPTIONS: Dict[int, str] = {
    TrafficAction.NO_ACTION: "No Intervention — Maintain baseline signal timing.",
    TrafficAction.EXTEND_GREEN: "Extend Green — Increase arterial green duration to clear queues.",
    TrafficAction.REDUCE_GREEN: "Reduce Green — Shorten arterial green to service cross-streets.",
    TrafficAction.TRAFFIC_DIVERSION: "Traffic Diversion — Reroute upstream traffic around congested zone.",
    TrafficAction.EMERGENCY_PRIORITY: "Emergency Priority — Preempt signals for active emergency vehicles.",
    TrafficAction.COMBINED_RESPONSE: "Combined Response — Simultaneous traffic diversion and emergency preemption.",
}

ACTION_DIM: int = len(TrafficAction)  # 6

ACTION_COSTS: Dict[int, float] = {
    TrafficAction.NO_ACTION: 0.0,
    TrafficAction.EXTEND_GREEN: 0.05,
    TrafficAction.REDUCE_GREEN: 0.05,
    TrafficAction.TRAFFIC_DIVERSION: 0.15,
    TrafficAction.EMERGENCY_PRIORITY: 0.10,
    TrafficAction.COMBINED_RESPONSE: 0.25,
}


# ---------------------------------------------------------------------------
# CityTrafficEnv
# ---------------------------------------------------------------------------

class CityTrafficEnv(gym.Env):
    """
    Gymnasium-compatible Urban Traffic Management Environment for Reinforcement Learning.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioType = ScenarioType.NORMAL,
        max_steps: int = 50,
        lstm_checkpoint: Optional[Union[str, Path]] = _ROOT / "models" / "best_lstm_model.pt",
        risk_checkpoint: Optional[Union[str, Path]] = _ROOT / "models" / "risk_model.pt",
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()

        self.scenario = scenario
        self.max_steps = max_steps
        self.current_step = 0
        self._seed = seed

        # Submodules
        self.sim = MockCitySimulator(seed=seed)
        self.gen = ScenarioGenerator()
        self.encoder = StateEncoder()

        # Load predictors if available
        self.lstm_predictor: Optional[TrafficPredictor] = None
        if lstm_checkpoint and Path(lstm_checkpoint).exists():
            self.lstm_predictor = TrafficPredictor.from_checkpoint(lstm_checkpoint)

        self.risk_predictor: Optional[RiskPredictor] = None
        if risk_checkpoint and Path(risk_checkpoint).exists():
            self.risk_predictor = RiskPredictor.from_checkpoint(risk_checkpoint)

        # Spaces
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        # History buffer for LSTM input (length = 10)
        self.history: List[CityState] = []
        self._last_state: Optional[CityState] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial 10-step history under the configured scenario.
        """
        if seed is not None:
            self._seed = seed

        # Optional scenario override from options
        if options and "scenario" in options:
            self.scenario = options["scenario"]

        self.sim.reset(seed=self._seed)
        self.current_step = 0

        overlay = self.gen.get_overlay(self.scenario)
        # Warmup history buffer with 10 initial steps
        self.history = [self.sim.step(overlay=overlay) for _ in range(10)]
        self._last_state = self.history[-1]

        state_vec = self._get_encoded_state()
        info = self._get_info()

        return state_vec, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: Union[int, TrafficAction],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one traffic control intervention action and observe the transition.
        """
        action = int(action)
        if not (0 <= action < ACTION_DIM):
            raise ValueError(f"Invalid action {action}. Must be in [0, {ACTION_DIM - 1}].")

        self.current_step += 1

        # 1. Compute overlay with action intervention effects
        base_overlay = self.gen.get_overlay(self.scenario)
        action_overlay = self._apply_action_effects(base_overlay, action)

        # 2. Advance simulation
        raw_next_state = self.sim.step(overlay=action_overlay)

        # 3. Post-process state with dynamic physical effects of intervention
        modified_state = self._postprocess_state_with_action(raw_next_state, action)
        self.history.append(modified_state)
        if len(self.history) > 10:
            self.history.pop(0)

        self._last_state = modified_state

        # 4. Compute reward
        reward = self._compute_reward(modified_state, action)

        # 5. Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # 6. Encode next state vector
        next_state_vec = self._get_encoded_state()
        info = self._get_info()
        info["action"] = action
        info["action_name"] = TrafficAction(action).name

        return next_state_vec, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _apply_action_effects(self, base_overlay: Dict, action: int) -> Dict:
        """Modify scenario ranges based on chosen control action."""
        overlay = dict(base_overlay)

        if action == TrafficAction.EXTEND_GREEN:
            # Helps traffic flow: speeds increase slightly, queue range decreases
            if "average_speed" in overlay:
                lo, hi = overlay["average_speed"]
                overlay["average_speed"] = (lo * 1.15, hi * 1.15)
            if "queue_length" in overlay:
                lo, hi = overlay["queue_length"]
                overlay["queue_length"] = (lo * 0.80, hi * 0.80)

        elif action == TrafficAction.REDUCE_GREEN:
            # Restricts flow on main corridor
            if "average_speed" in overlay:
                lo, hi = overlay["average_speed"]
                overlay["average_speed"] = (lo * 0.85, hi * 0.85)
            if "queue_length" in overlay:
                lo, hi = overlay["queue_length"]
                overlay["queue_length"] = (lo * 1.25, min(1000.0, hi * 1.25))

        elif action == TrafficAction.TRAFFIC_DIVERSION:
            # Diverts vehicles: vehicle count and density reduce
            if "vehicle_count" in overlay:
                lo, hi = overlay["vehicle_count"]
                overlay["vehicle_count"] = (int(lo * 0.75), int(hi * 0.75))
            if "traffic_density" in overlay:
                lo, hi = overlay["traffic_density"]
                overlay["traffic_density"] = (lo * 0.75, hi * 0.75)
            if "queue_length" in overlay:
                lo, hi = overlay["queue_length"]
                overlay["queue_length"] = (lo * 0.70, hi * 0.70)

        elif action == TrafficAction.EMERGENCY_PRIORITY:
            # Preemption clears emergency vehicles faster
            overlay["traffic_signal_state"] = "green"

        elif action == TrafficAction.COMBINED_RESPONSE:
            # Strongest relief: diversion + preemption
            if "vehicle_count" in overlay:
                lo, hi = overlay["vehicle_count"]
                overlay["vehicle_count"] = (int(lo * 0.70), int(hi * 0.70))
            if "traffic_density" in overlay:
                lo, hi = overlay["traffic_density"]
                overlay["traffic_density"] = (lo * 0.70, hi * 0.70)
            if "queue_length" in overlay:
                lo, hi = overlay["queue_length"]
                overlay["queue_length"] = (lo * 0.60, hi * 0.60)
            overlay["traffic_signal_state"] = "green"

        return overlay

    def _postprocess_state_with_action(self, state: CityState, action: int) -> CityState:
        """Apply bounded post-processing adjustments to the state dataclass."""
        ev_count = state.emergency_vehicle_count
        if action in (TrafficAction.EMERGENCY_PRIORITY, TrafficAction.COMBINED_RESPONSE) and ev_count > 0:
            # Emergency vehicles clear faster under preemption
            ev_count = max(0, ev_count - 1)
            state.emergency_vehicle_count = ev_count

        return state

    def _get_encoded_state(self) -> np.ndarray:
        """Chain LSTM prediction and Risk estimation to produce the 20-dim state."""
        current_state = self.history[-1]

        # 1. Forecast from LSTM
        if self.lstm_predictor is not None:
            forecast = self.lstm_predictor.predict(self.history)
        else:
            forecast = {
                "traffic_density": [current_state.traffic_density] * 5,
                "average_speed": [current_state.average_speed] * 5,
                "queue_length": [current_state.queue_length] * 5,
            }

        # 2. Risk from RiskPredictor
        if self.risk_predictor is not None:
            risk_res = self.risk_predictor.predict(current_state, forecast)
            risk_score = float(risk_res["risk_score"])
        else:
            risk_score = 50.0

        # 3. Encode into 20-dim state
        return self.encoder.encode(current_state, forecast, risk_score)

    def _compute_reward(self, state: CityState, action: int) -> float:
        """
        Multi-objective reward function.
        """
        # Normalized performance components
        density_reward = 1.0 - float(np.clip(state.traffic_density, 0.0, 1.0))
        speed_reward = float(np.clip(state.average_speed / 60.0, 0.0, 1.0))
        queue_reward = 1.0 - float(np.clip(state.queue_length / 500.0, 0.0, 1.0))

        # Emergency responsiveness bonus / penalty
        ev_count = state.emergency_vehicle_count
        emergency_reward = 0.0
        if ev_count > 0:
            if action in (TrafficAction.EMERGENCY_PRIORITY, TrafficAction.COMBINED_RESPONSE):
                emergency_reward = 1.0 + 0.5 * ev_count
            else:
                emergency_reward = -0.8 * ev_count

        # Action cost penalty
        action_cost = ACTION_COSTS.get(action, 0.0)

        # Total reward
        r = (
            1.0 * density_reward
            + 0.8 * speed_reward
            + 0.8 * queue_reward
            + emergency_reward
            - action_cost
        )
        return float(r)

    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        current_state = self.history[-1]
        return {
            "timestep": self.current_step,
            "vehicle_count": current_state.vehicle_count,
            "average_speed": current_state.average_speed,
            "traffic_density": current_state.traffic_density,
            "queue_length": current_state.queue_length,
            "emergency_vehicle_count": current_state.emergency_vehicle_count,
            "accident": current_state.accident,
            "road_blockage": current_state.road_blockage,
            "scenario": self.scenario.name,
        }
