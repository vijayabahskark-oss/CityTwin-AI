from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import carla

from simulation.mock_city import CityState


class CarlaStateAdapter:
    """
    Converts CARLA world observations into the existing CityState format.

    This adapter does not modify CityState or MockCitySimulator.
    It provides a CARLA-backed source of observations for the
    existing CityTwin ML pipeline.
    """

    MAX_VEHICLES = 500
    MAX_PEDESTRIANS = 300

    MAX_SPEED_KMH = 120.0
    MAX_QUEUE_LENGTH_M = 1000.0

    # Used to normalize the number of observed vehicles.
    # This is an MVP simulation normalization, not a calibrated
    # real-world traffic-density measurement.
    DENSITY_REFERENCE_VEHICLES = 200

    QUEUE_SPEED_THRESHOLD_KMH = 5.0

    def __init__(
        self,
        world: carla.World,
        scenario_tag: str = "carla_normal",
    ) -> None:
        self.world = world
        self.scenario_tag = scenario_tag

    def get_state(
        self,
        timestep: Optional[int] = None,
        accident: bool = False,
        road_blockage: bool = False,
        emergency_vehicle_count: Optional[int] = None,
    ) -> CityState:
        """
        Read the current CARLA world and return a CityState.
        """

        vehicles = list(
            self.world.get_actors().filter("vehicle.*")
        )

        pedestrians = list(
            self.world.get_actors().filter("walker.pedestrian.*")
        )

        traffic_lights = list(
            self.world.get_actors().filter("traffic.traffic_light")
        )

        vehicle_count = min(
            len(vehicles),
            self.MAX_VEHICLES,
        )

        pedestrian_count = min(
            len(pedestrians),
            self.MAX_PEDESTRIANS,
        )

        # ---------------------------------------------------------
        # Vehicle speeds
        # ---------------------------------------------------------

        speeds = []

        for vehicle in vehicles:

            velocity = vehicle.get_velocity()

            speed_ms = (
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            ) ** 0.5

            speed_kmh = speed_ms * 3.6

            speeds.append(
                min(speed_kmh, self.MAX_SPEED_KMH)
            )

        if speeds:
            average_speed = sum(speeds) / len(speeds)
        else:
            average_speed = 0.0

        average_speed = max(
            0.0,
            min(
                self.MAX_SPEED_KMH,
                average_speed,
            ),
        )

        # ---------------------------------------------------------
        # Traffic density
        # ---------------------------------------------------------

        traffic_density = min(
            1.0,
            vehicle_count / self.DENSITY_REFERENCE_VEHICLES,
        )

        # ---------------------------------------------------------
        # Queue estimation
        # ---------------------------------------------------------

        slow_vehicle_count = sum(
            1
            for speed in speeds
            if speed <= self.QUEUE_SPEED_THRESHOLD_KMH
        )

        # MVP approximation:
        # each slow/stopped vehicle contributes approximately
        # 7 metres to the queue.
        queue_length = min(
            self.MAX_QUEUE_LENGTH_M,
            slow_vehicle_count * 7.0,
        )

        # ---------------------------------------------------------
        # Weather
        # ---------------------------------------------------------

        weather = self.world.get_weather()

        rain_intensity = max(
            0.0,
            min(
                1.0,
                weather.precipitation / 100.0,
            ),
        )

        fog_intensity = max(
            0.0,
            min(
                1.0,
                weather.fog_density / 100.0,
            ),
        )

        # CARLA weather temperature is not directly available
        # as a physical sensor value, so use a neutral approximation
        # for the initial adapter.
        temperature = 25.0

        # ---------------------------------------------------------
        # Emergency vehicles
        # ---------------------------------------------------------

        if emergency_vehicle_count is not None:

            ev_count = emergency_vehicle_count

        else:

            ev_count = 0

            for vehicle in vehicles:

                type_id = vehicle.type_id.lower()

                if any(
                    keyword in type_id
                    for keyword in (
                        "ambulance",
                        "firetruck",
                        "police",
                    )
                ):
                    ev_count += 1

        ev_count = max(
            0,
            min(10, ev_count),
        )

        # ---------------------------------------------------------
        # Traffic signal
        # ---------------------------------------------------------

        traffic_signal_state = self._get_signal_state(
            traffic_lights
        )

        # ---------------------------------------------------------
        # Timestamp
        # ---------------------------------------------------------

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        if timestep is None:
            timestep = 0

        # ---------------------------------------------------------
        # Build CityState
        # ---------------------------------------------------------

        state = CityState(
            vehicle_count=vehicle_count,
            average_speed=round(
                average_speed,
                2,
            ),
            traffic_density=round(
                traffic_density,
                4,
            ),
            queue_length=round(
                queue_length,
                2,
            ),
            pedestrian_count=pedestrian_count,
            rain_intensity=round(
                rain_intensity,
                4,
            ),
            fog_intensity=round(
                fog_intensity,
                4,
            ),
            temperature=temperature,
            road_blockage=bool(
                road_blockage
            ),
            accident=bool(
                accident
            ),
            emergency_vehicle_count=ev_count,
            traffic_signal_state=traffic_signal_state,
            timestamp=timestamp,
            timestep=int(timestep),
            scenario_tag=self.scenario_tag,
        )

        state.validate()

        return state

    @staticmethod
    def _get_signal_state(
        traffic_lights,
    ) -> str:
        """
        Convert currently observed CARLA traffic-light states
        into the existing CityState categorical representation.
        """

        if not traffic_lights:
            return "flashing"

        # Use the first observed light for the MVP.
        state = traffic_lights[0].get_state()

        if state == carla.TrafficLightState.Green:
            return "green"

        if state == carla.TrafficLightState.Yellow:
            return "yellow"

        if state == carla.TrafficLightState.Red:
            return "red"

        return "flashing"