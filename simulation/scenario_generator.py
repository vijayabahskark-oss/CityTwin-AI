"""
scenario_generator.py — ScenarioGenerator
==========================================
Defines a catalogue of urban traffic scenarios as *overlays* that are
injected into MockCitySimulator.step() to shift its baseline distributions.

Each scenario is a pure data description (no simulation logic lives here).
This separation keeps scenarios declarative and easy to extend without
touching simulator internals.

Scenarios implemented
---------------------
normal           — ordinary daytime traffic
rush_hour        — peak-hour congestion
accident         — active collision on the road
heavy_rain       — severe rainfall
heavy_rain_fog   — rain + reduced visibility
accident_rain    — accident in adverse weather
road_blockage    — construction / debris blocking lanes
emergency        — emergency vehicle response

Usage
-----
>>> from simulation import ScenarioGenerator, MockCitySimulator, ScenarioType
>>> gen = ScenarioGenerator()
>>> sim = MockCitySimulator(seed=7)
>>> states = sim.run(steps=20, overlay=gen.get_overlay(ScenarioType.RUSH_HOUR))
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import random


# ---------------------------------------------------------------------------
# ScenarioType enum
# ---------------------------------------------------------------------------

class ScenarioType(Enum):
    """All first-class scenario labels understood by ScenarioGenerator."""
    NORMAL          = auto()
    RUSH_HOUR       = auto()
    ACCIDENT        = auto()
    HEAVY_RAIN      = auto()
    HEAVY_RAIN_FOG  = auto()
    ACCIDENT_RAIN   = auto()
    ROAD_BLOCKAGE   = auto()
    EMERGENCY       = auto()


# ---------------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Provides overlay dictionaries for each ScenarioType.

    An *overlay* is a plain dict whose keys are CityState field names
    and whose values are either:
      - ``(min, max)`` tuples  — replace the simulator's baseline range
      - scalar value           — set the field to an exact value

    Parameters
    ----------
    seed : int | None
        Optional seed for any stochastic variation within scenario
        generation itself (currently unused but reserved for future use).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_overlay(self, scenario: ScenarioType) -> Dict:
        """
        Return the parameter overlay dict for the requested scenario.

        Parameters
        ----------
        scenario : ScenarioType
            The scenario whose overlay you want.

        Returns
        -------
        dict
            Overlay ready to pass to ``MockCitySimulator.step()`` or
            ``MockCitySimulator.run()``.
        """
        _dispatch = {
            ScenarioType.NORMAL:         self._normal,
            ScenarioType.RUSH_HOUR:      self._rush_hour,
            ScenarioType.ACCIDENT:       self._accident,
            ScenarioType.HEAVY_RAIN:     self._heavy_rain,
            ScenarioType.HEAVY_RAIN_FOG: self._heavy_rain_fog,
            ScenarioType.ACCIDENT_RAIN:  self._accident_rain,
            ScenarioType.ROAD_BLOCKAGE:  self._road_blockage,
            ScenarioType.EMERGENCY:      self._emergency,
        }
        handler = _dispatch.get(scenario)
        if handler is None:
            raise ValueError(f"Unknown scenario: {scenario!r}")
        return handler()

    def all_scenarios(self) -> List[ScenarioType]:
        """Return a list of every available ScenarioType."""
        return list(ScenarioType)

    def describe(self, scenario: ScenarioType) -> str:
        """Return a human-readable description of a scenario."""
        _descriptions = {
            ScenarioType.NORMAL: (
                "Normal daytime traffic — moderate vehicle count, "
                "free-flowing speeds, no incidents."
            ),
            ScenarioType.RUSH_HOUR: (
                "Peak-hour congestion — high vehicle density, slow speeds, "
                "long queues, many pedestrians."
            ),
            ScenarioType.ACCIDENT: (
                "Active road accident — speed halved, queues doubled, "
                "emergency vehicles on scene."
            ),
            ScenarioType.HEAVY_RAIN: (
                "Severe rainfall — reduced speed, slight increase in incidents, "
                "low fog."
            ),
            ScenarioType.HEAVY_RAIN_FOG: (
                "Rain combined with dense fog — severely impaired visibility, "
                "very low speeds."
            ),
            ScenarioType.ACCIDENT_RAIN: (
                "Accident occurring during heavy rain — worst-case combined "
                "traffic disruption."
            ),
            ScenarioType.ROAD_BLOCKAGE: (
                "Lane / road blocked by construction or debris — "
                "high queues, reduced capacity."
            ),
            ScenarioType.EMERGENCY: (
                "Emergency vehicle response — multiple EVs, flashing signals, "
                "traffic yielding."
            ),
        }
        return _descriptions.get(scenario, "No description available.")

    # ------------------------------------------------------------------
    # Private overlay factories
    # ------------------------------------------------------------------

    @staticmethod
    def _normal() -> Dict:
        return {
            "vehicle_count":           (80, 180),
            "average_speed":           (35.0, 60.0),
            "traffic_density":         (0.15, 0.40),
            "queue_length":            (10.0, 120.0),
            "pedestrian_count":        (20, 80),
            "rain_intensity":          (0.0, 0.05),
            "fog_intensity":           (0.0, 0.03),
            "temperature":             (18.0, 28.0),
            "road_blockage":           False,
            "accident":                False,
            "emergency_vehicle_count": (0, 0),
            "scenario_tag":            ScenarioType.NORMAL.name.lower(),
        }

    @staticmethod
    def _rush_hour() -> Dict:
        return {
            "vehicle_count":           (320, 480),
            "average_speed":           (8.0, 25.0),
            "traffic_density":         (0.70, 0.95),
            "queue_length":            (250.0, 700.0),
            "pedestrian_count":        (150, 280),
            "rain_intensity":          (0.0, 0.08),
            "fog_intensity":           (0.0, 0.05),
            "temperature":             (18.0, 28.0),
            "road_blockage":           False,
            "accident":                False,
            "emergency_vehicle_count": (0, 0),
            "scenario_tag":            ScenarioType.RUSH_HOUR.name.lower(),
        }

    @staticmethod
    def _accident() -> Dict:
        return {
            "vehicle_count":           (200, 380),
            "average_speed":           (5.0, 20.0),
            "traffic_density":         (0.60, 0.90),
            "queue_length":            (300.0, 900.0),
            "pedestrian_count":        (30, 100),
            "rain_intensity":          (0.0, 0.10),
            "fog_intensity":           (0.0, 0.05),
            "temperature":             (15.0, 30.0),
            "road_blockage":           True,
            "accident":                True,
            "emergency_vehicle_count": (1, 4),
            "scenario_tag":            ScenarioType.ACCIDENT.name.lower(),
        }

    @staticmethod
    def _heavy_rain() -> Dict:
        return {
            "vehicle_count":           (100, 250),
            "average_speed":           (15.0, 40.0),
            "traffic_density":         (0.30, 0.65),
            "queue_length":            (80.0, 350.0),
            "pedestrian_count":        (5, 40),
            "rain_intensity":          (0.65, 0.95),
            "fog_intensity":           (0.05, 0.20),
            "temperature":             (10.0, 22.0),
            "road_blockage":           False,
            "accident":                False,
            "emergency_vehicle_count": (0, 1),
            "scenario_tag":            ScenarioType.HEAVY_RAIN.name.lower(),
        }

    @staticmethod
    def _heavy_rain_fog() -> Dict:
        return {
            "vehicle_count":           (60, 180),
            "average_speed":           (5.0, 20.0),
            "traffic_density":         (0.25, 0.60),
            "queue_length":            (150.0, 600.0),
            "pedestrian_count":        (2, 20),
            "rain_intensity":          (0.60, 0.90),
            "fog_intensity":           (0.55, 0.95),
            "temperature":             (8.0, 18.0),
            "road_blockage":           False,
            "accident":                False,
            "emergency_vehicle_count": (0, 1),
            "scenario_tag":            ScenarioType.HEAVY_RAIN_FOG.name.lower(),
        }

    @staticmethod
    def _accident_rain() -> Dict:
        return {
            "vehicle_count":           (180, 360),
            "average_speed":           (3.0, 15.0),
            "traffic_density":         (0.70, 0.98),
            "queue_length":            (400.0, 980.0),
            "pedestrian_count":        (5, 50),
            "rain_intensity":          (0.55, 0.90),
            "fog_intensity":           (0.10, 0.35),
            "temperature":             (8.0, 20.0),
            "road_blockage":           True,
            "accident":                True,
            "emergency_vehicle_count": (2, 6),
            "scenario_tag":            ScenarioType.ACCIDENT_RAIN.name.lower(),
        }

    @staticmethod
    def _road_blockage() -> Dict:
        return {
            "vehicle_count":           (180, 380),
            "average_speed":           (3.0, 18.0),
            "traffic_density":         (0.60, 0.92),
            "queue_length":            (350.0, 950.0),
            "pedestrian_count":        (20, 80),
            "rain_intensity":          (0.0, 0.10),
            "fog_intensity":           (0.0, 0.08),
            "temperature":             (12.0, 30.0),
            "road_blockage":           True,
            "accident":                False,
            "emergency_vehicle_count": (0, 2),
            "scenario_tag":            ScenarioType.ROAD_BLOCKAGE.name.lower(),
        }

    @staticmethod
    def _emergency() -> Dict:
        return {
            "vehicle_count":           (150, 350),
            "average_speed":           (10.0, 35.0),
            "traffic_density":         (0.40, 0.80),
            "queue_length":            (100.0, 500.0),
            "pedestrian_count":        (10, 60),
            "rain_intensity":          (0.0, 0.15),
            "fog_intensity":           (0.0, 0.10),
            "temperature":             (12.0, 30.0),
            "road_blockage":           False,
            "accident":                False,
            "emergency_vehicle_count": (3, 8),
            "traffic_signal_state":    "flashing",
            "scenario_tag":            ScenarioType.EMERGENCY.name.lower(),
        }
