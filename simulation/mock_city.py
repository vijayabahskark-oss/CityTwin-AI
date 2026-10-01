"""
mock_city.py — MockCitySimulator
=================================
Generates realistic, timestep-based urban traffic states without
requiring CARLA, a real sensor network, or any ML model.

Each call to `step()` advances the simulation by one timestep and
returns a CityState dataclass that the rest of CityTwin AI will
consume (CNN feature extraction, LSTM sequence modelling, RL control).

Design principles
-----------------
* Reproducible  — pass a fixed `seed` to get the same run every time.
* Self-contained — zero external dependencies beyond the stdlib.
* Modular       — ScenarioGenerator injects *overlays* that bias the
                  base distributions; no hard-coded if-else chains here.
* Validated     — CityState.validate() raises ValueError for impossible
                  physics so bad data never silently enters the pipeline.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CityState — one timestep snapshot
# ---------------------------------------------------------------------------

@dataclass
class CityState:
    """
    A single timestep snapshot of the simulated city grid.

    All numeric fields use SI-like units or normalised [0, 1] scales
    as documented per-field below.
    """

    # --- Traffic flow ---
    vehicle_count: int          # Total vehicles in monitored zone [0, 500]
    average_speed: float        # km/h, meaningful range [0, 120]
    traffic_density: float      # vehicles / km² , normalised [0, 1]
    queue_length: float         # metres of longest queue  [0, 1 000]

    # --- Pedestrians ---
    pedestrian_count: int       # Pedestrians in zone [0, 300]

    # --- Environment ---
    rain_intensity: float       # 0 = dry, 1 = extreme downpour
    fog_intensity: float        # 0 = clear, 1 = zero visibility
    temperature: float          # °C

    # --- Incidents ---
    road_blockage: bool         # Lane / road blocked
    accident: bool              # Active accident
    emergency_vehicle_count: int  # Fire / ambulance / police [0, 10]

    # --- Infrastructure ---
    traffic_signal_state: str   # "green" | "yellow" | "red" | "flashing"

    # --- Meta ---
    timestamp: str              # ISO-8601 UTC
    timestep: int               # Simulation tick index
    scenario_tag: str = "normal"  # Label for ML training

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Raise ValueError if any field violates physical / domain constraints.
        Called automatically by MockCitySimulator after each step.
        """
        errors: List[str] = []

        if not (0 <= self.vehicle_count <= 500):
            errors.append(f"vehicle_count={self.vehicle_count} out of [0,500]")
        if not (0.0 <= self.average_speed <= 120.0):
            errors.append(f"average_speed={self.average_speed:.2f} out of [0,120]")
        if not (0.0 <= self.traffic_density <= 1.0):
            errors.append(f"traffic_density={self.traffic_density:.3f} out of [0,1]")
        if not (0.0 <= self.queue_length <= 1000.0):
            errors.append(f"queue_length={self.queue_length:.1f} out of [0,1000]")
        if not (0 <= self.pedestrian_count <= 300):
            errors.append(f"pedestrian_count={self.pedestrian_count} out of [0,300]")
        if not (0.0 <= self.rain_intensity <= 1.0):
            errors.append(f"rain_intensity={self.rain_intensity:.3f} out of [0,1]")
        if not (0.0 <= self.fog_intensity <= 1.0):
            errors.append(f"fog_intensity={self.fog_intensity:.3f} out of [0,1]")
        if not (-50.0 <= self.temperature <= 60.0):
            errors.append(f"temperature={self.temperature:.1f} out of [-50,60]")
        if not (0 <= self.emergency_vehicle_count <= 10):
            errors.append(
                f"emergency_vehicle_count={self.emergency_vehicle_count} out of [0,10]"
            )
        valid_signals = {"green", "yellow", "red", "flashing"}
        if self.traffic_signal_state not in valid_signals:
            errors.append(
                f"traffic_signal_state='{self.traffic_signal_state}' "
                f"not in {valid_signals}"
            )

        if errors:
            raise ValueError("CityState validation failed:\n  " + "\n  ".join(errors))

    def to_dict(self) -> Dict:
        """Return a plain dictionary (useful for JSON serialisation)."""
        return asdict(self)

    def __str__(self) -> str:
        inc_flags = []
        if self.accident:
            inc_flags.append("ACCIDENT")
        if self.road_blockage:
            inc_flags.append("BLOCKAGE")
        if self.emergency_vehicle_count > 0:
            inc_flags.append(f"EV×{self.emergency_vehicle_count}")
        incident_str = ", ".join(inc_flags) if inc_flags else "none"

        return (
            f"[t={self.timestep:04d} | {self.scenario_tag:22s}] "
            f"vehicles={self.vehicle_count:3d}  "
            f"speed={self.average_speed:5.1f}km/h  "
            f"density={self.traffic_density:.3f}  "
            f"queue={self.queue_length:6.1f}m  "
            f"peds={self.pedestrian_count:3d}  "
            f"rain={self.rain_intensity:.2f}  "
            f"fog={self.fog_intensity:.2f}  "
            f"signal={self.traffic_signal_state:8s}  "
            f"incidents=[{incident_str}]"
        )


# ---------------------------------------------------------------------------
# MockCitySimulator
# ---------------------------------------------------------------------------

class MockCitySimulator:
    """
    Produces a sequence of CityState objects representing urban traffic
    dynamics under configurable scenario overlays.

    Parameters
    ----------
    seed : int | None
        Random seed for full reproducibility. ``None`` means non-deterministic.
    validate : bool
        If True (default) every generated state is validated before being
        returned; raises ValueError on constraint violations.
    base_timestep_seconds : int
        Simulated wall-clock seconds each tick represents (default 30 s).

    Example
    -------
    >>> sim = MockCitySimulator(seed=42)
    >>> for state in sim.run(steps=5):
    ...     print(state)
    """

    # Baseline parameter ranges — scenario overlays shift these
    _BASELINE = {
        "vehicle_count":          (80, 180),    # uniform range
        "average_speed":          (35.0, 60.0), # km/h
        "traffic_density":        (0.15, 0.40),
        "queue_length":           (10.0, 120.0),
        "pedestrian_count":       (20, 80),
        "rain_intensity":         (0.0, 0.05),
        "fog_intensity":          (0.0, 0.03),
        "temperature":            (18.0, 28.0),
        "emergency_vehicle_count": (0, 0),
    }

    _SIGNAL_CYCLE = ["green", "green", "green", "yellow", "red", "red", "red", "red"]

    def __init__(
        self,
        seed: Optional[int] = 42,
        validate: bool = True,
        base_timestep_seconds: int = 30,
    ) -> None:
        self.seed = seed
        self._validate = validate
        self.base_timestep_seconds = base_timestep_seconds

        self._rng = random.Random(seed)
        self._tick: int = 0
        self._signal_index: int = 0
        self._sim_epoch: float = time.time()  # wall-clock anchor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the simulator to tick 0, optionally with a new seed."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._tick = 0
        self._signal_index = 0
        self._sim_epoch = time.time()

    def step(self, overlay: Optional[Dict] = None) -> CityState:
        """
        Advance the simulation by one tick and return a CityState.

        Parameters
        ----------
        overlay : dict | None
            Parameter overrides injected by ScenarioGenerator.
            Keys match CityState field names; values are
            ``(min, max)`` tuples to replace the baseline range,
            or scalar values for boolean / categorical fields.
        """
        overlay = overlay or {}
        params = {**self._BASELINE}

        # Apply overlay — merge ranges or replace scalars
        for key, val in overlay.items():
            params[key] = val

        def _sample_range(key: str) -> float:
            lo, hi = params[key]
            return self._rng.uniform(lo, hi)

        def _sample_int_range(key: str) -> int:
            lo, hi = params[key]
            return self._rng.randint(int(lo), int(hi))

        # --- Vehicle dynamics ---
        vehicle_count = _sample_int_range("vehicle_count")
        average_speed = round(_sample_range("average_speed"), 2)
        traffic_density = round(min(1.0, _sample_range("traffic_density")), 4)

        # Queue length correlates with speed: slower → longer queues
        base_queue = _sample_range("queue_length")
        speed_factor = max(0.0, (60.0 - average_speed) / 60.0)
        queue_length = round(min(1000.0, base_queue * (1.0 + speed_factor)), 2)

        # --- Pedestrians ---
        pedestrian_count = _sample_int_range("pedestrian_count")

        # --- Environment ---
        rain_intensity = round(min(1.0, max(0.0, _sample_range("rain_intensity"))), 4)
        fog_intensity = round(min(1.0, max(0.0, _sample_range("fog_intensity"))), 4)
        temperature = round(_sample_range("temperature"), 1)

        # Rain slightly reduces speed
        if rain_intensity > 0.3:
            average_speed = round(max(0.0, average_speed * (1.0 - rain_intensity * 0.3)), 2)

        # Fog significantly reduces speed
        if fog_intensity > 0.4:
            average_speed = round(max(0.0, average_speed * (1.0 - fog_intensity * 0.5)), 2)

        # --- Incidents ---
        road_blockage = bool(params.get("road_blockage", False))
        accident = bool(params.get("accident", False))
        ev_lo, ev_hi = params.get("emergency_vehicle_count", (0, 0))
        emergency_vehicle_count = self._rng.randint(int(ev_lo), int(ev_hi))

        # Blockages / accidents reduce speed further
        if road_blockage or accident:
            average_speed = round(max(0.0, average_speed * 0.5), 2)
            queue_length = round(min(1000.0, queue_length * 2.0), 2)
            traffic_density = round(min(1.0, traffic_density * 1.4), 4)

        # --- Traffic signal ---
        signal_override = params.get("traffic_signal_state", None)
        if signal_override:
            traffic_signal_state = signal_override
        else:
            traffic_signal_state = self._SIGNAL_CYCLE[
                self._signal_index % len(self._SIGNAL_CYCLE)
            ]
        self._signal_index += 1

        # --- Timestamp (simulated wall-clock) ---
        sim_seconds = self._tick * self.base_timestep_seconds
        sim_time = datetime.fromtimestamp(
            self._sim_epoch + sim_seconds, tz=timezone.utc
        )
        timestamp = sim_time.isoformat()

        # --- Scenario tag ---
        scenario_tag = str(params.get("scenario_tag", "normal"))

        state = CityState(
            vehicle_count=vehicle_count,
            average_speed=average_speed,
            traffic_density=traffic_density,
            queue_length=queue_length,
            pedestrian_count=pedestrian_count,
            rain_intensity=rain_intensity,
            fog_intensity=fog_intensity,
            temperature=temperature,
            road_blockage=road_blockage,
            accident=accident,
            emergency_vehicle_count=emergency_vehicle_count,
            traffic_signal_state=traffic_signal_state,
            timestamp=timestamp,
            timestep=self._tick,
            scenario_tag=scenario_tag,
        )

        if self._validate:
            state.validate()

        self._tick += 1
        return state

    def run(
        self,
        steps: int = 100,
        overlay: Optional[Dict] = None,
    ) -> List[CityState]:
        """
        Run the simulator for `steps` ticks, returning the full sequence.

        Parameters
        ----------
        steps : int
            Number of timesteps to generate.
        overlay : dict | None
            Static overlay applied to every tick (see ``step()``).
        """
        return [self.step(overlay=overlay) for _ in range(steps)]
