import sys
import time
import random
import carla

sys.path.insert(0, ".")

from carla_integration.state_adapter import CarlaStateAdapter


# ============================================================
# CONFIGURATION
# ============================================================

SCENARIOS = {
    1: "NORMAL",
    2: "RUSH_HOUR",
    3: "HEAVY_RAIN",
    4: "FOG",
    5: "ACCIDENT",
    6: "ROAD_BLOCKAGE",
    7: "EMERGENCY",
    8: "HEAVY_RAIN_ACCIDENT",
}

SCENARIO_VEHICLE_COUNT = {
    "NORMAL": 10,
    "RUSH_HOUR": 30,
    "HEAVY_RAIN": 15,
    "FOG": 15,
    "ACCIDENT": 15,
    "ROAD_BLOCKAGE": 15,
    "EMERGENCY": 15,
    "HEAVY_RAIN_ACCIDENT": 20,
}

SETTLE_TIME = 5.0


# ============================================================
# CLEAN ALL EXISTING VEHICLES
# ============================================================

def cleanup_existing_traffic(world):

    print("\nCleaning existing CARLA traffic...")

    actors = world.get_actors()

    vehicles = actors.filter(
        "vehicle.*"
    )

    destroyed = 0

    for vehicle in vehicles:

        try:

            vehicle.set_autopilot(
                False
            )

            vehicle.destroy()

            destroyed += 1

        except Exception:
            pass

    print(
        f"Removed existing vehicles: "
        f"{destroyed}"
    )

    # Give CARLA time to remove destroyed actors.
    time.sleep(1.0)


# ============================================================
# SPAWN VEHICLES
# ============================================================

def spawn_vehicles(
    world,
    number
):

    blueprint_library = (
        world.get_blueprint_library()
    )

    vehicle_blueprints = (
        blueprint_library.filter(
            "vehicle.*"
        )
    )

    spawn_points = (
        world.get_map().get_spawn_points()
    )

    random.shuffle(
        spawn_points
    )

    vehicles = []

    for spawn_point in spawn_points:

        if len(vehicles) >= number:
            break

        blueprint = random.choice(
            vehicle_blueprints
        )

        if blueprint.has_attribute(
            "role_name"
        ):

            blueprint.set_attribute(
                "role_name",
                "citytwin"
            )

        vehicle = world.try_spawn_actor(
            blueprint,
            spawn_point
        )

        if vehicle is not None:

            vehicles.append(
                vehicle
            )

    return vehicles


# ============================================================
# ENABLE AUTOPILOT
# ============================================================

def enable_autopilot(
    vehicles,
    client
):

    traffic_manager = (
        client.get_trafficmanager(
            8000
        )
    )

    traffic_manager.set_global_distance_to_leading_vehicle(
        5.0
    )

    for vehicle in vehicles:

        try:

            vehicle.set_autopilot(
                True,
                traffic_manager.get_port()
            )

        except Exception:
            pass

    print(
        "Traffic Manager autopilot enabled."
    )


# ============================================================
# WEATHER
# ============================================================

def apply_weather(
    world,
    scenario
):

    weather = (
        carla.WeatherParameters()
    )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if scenario == "NORMAL":

        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 0.0

    # --------------------------------------------------------
    # RUSH HOUR
    # --------------------------------------------------------

    elif scenario == "RUSH_HOUR":

        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 0.0

    # --------------------------------------------------------
    # HEAVY RAIN
    # --------------------------------------------------------

    elif scenario == "HEAVY_RAIN":

        weather.cloudiness = 80.0
        weather.precipitation = 90.0
        weather.precipitation_deposits = 80.0
        weather.wind_intensity = 40.0
        weather.fog_density = 15.0
        weather.wetness = 90.0

    # --------------------------------------------------------
    # FOG
    # --------------------------------------------------------

    elif scenario == "FOG":

        weather.cloudiness = 70.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 85.0
        weather.wetness = 20.0
        weather.wind_intensity = 5.0

    # --------------------------------------------------------
    # ACCIDENT
    # --------------------------------------------------------

    elif scenario == "ACCIDENT":

        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 0.0

    # --------------------------------------------------------
    # ROAD BLOCKAGE
    # --------------------------------------------------------

    elif scenario == "ROAD_BLOCKAGE":

        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 0.0

    # --------------------------------------------------------
    # EMERGENCY
    # --------------------------------------------------------

    elif scenario == "EMERGENCY":

        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 0.0

    # --------------------------------------------------------
    # HEAVY RAIN + ACCIDENT
    # --------------------------------------------------------

    elif scenario == "HEAVY_RAIN_ACCIDENT":

        weather.cloudiness = 90.0
        weather.precipitation = 95.0
        weather.precipitation_deposits = 90.0
        weather.wind_intensity = 50.0
        weather.fog_density = 30.0
        weather.wetness = 95.0

    else:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    world.set_weather(
        weather
    )


# ============================================================
# SCENARIO FLAGS
# ============================================================

def get_scenario_flags(
    scenario
):

    accident = False
    road_blockage = False
    emergency_vehicle_count = 0

    if scenario == "ACCIDENT":

        accident = True

    elif scenario == "ROAD_BLOCKAGE":

        road_blockage = True

    elif scenario == "EMERGENCY":

        emergency_vehicle_count = 1

    elif scenario == "HEAVY_RAIN_ACCIDENT":

        accident = True

    return (
        accident,
        road_blockage,
        emergency_vehicle_count
    )


# ============================================================
# PRINT STATE
# ============================================================

def print_state(
    state
):

    print("\n" + "-" * 70)

    print(
        f"Vehicle count       : "
        f"{state.vehicle_count}"
    )

    print(
        f"Average speed       : "
        f"{state.average_speed:.2f} km/h"
    )

    print(
        f"Traffic density     : "
        f"{state.traffic_density:.3f}"
    )

    print(
        f"Queue length        : "
        f"{state.queue_length:.2f} m"
    )

    print(
        f"Pedestrians         : "
        f"{state.pedestrian_count}"
    )

    print(
        f"Rain intensity      : "
        f"{state.rain_intensity:.3f}"
    )

    print(
        f"Fog intensity       : "
        f"{state.fog_intensity:.3f}"
    )

    print(
        f"Accident            : "
        f"{state.accident}"
    )

    print(
        f"Road blockage       : "
        f"{state.road_blockage}"
    )

    print(
        f"Emergency vehicles  : "
        f"{state.emergency_vehicle_count}"
    )

    print(
        f"Traffic signal      : "
        f"{state.traffic_signal_state}"
    )

    print("-" * 70)


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_scenario(
    world,
    client,
    scenario
):

    print("\n\n")

    print("=" * 70)

    print(
        f"SCENARIO: {scenario}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. CLEAN PREVIOUS TRAFFIC
    # --------------------------------------------------------

    cleanup_existing_traffic(
        world
    )

    # --------------------------------------------------------
    # 2. APPLY WEATHER
    # --------------------------------------------------------

    print(
        f"\nApplying scenario weather..."
    )

    apply_weather(
        world,
        scenario
    )

    # --------------------------------------------------------
    # 3. SPAWN CONTROLLED TRAFFIC
    # --------------------------------------------------------

    target_count = (
        SCENARIO_VEHICLE_COUNT[
            scenario
        ]
    )

    print(
        f"Target vehicles: "
        f"{target_count}"
    )

    vehicles = spawn_vehicles(
        world,
        target_count
    )

    print(
        f"Spawned vehicles: "
        f"{len(vehicles)}"
    )

    # --------------------------------------------------------
    # 4. AUTOPILOT
    # --------------------------------------------------------

    enable_autopilot(
        vehicles,
        client
    )

    # --------------------------------------------------------
    # 5. ALLOW TRAFFIC TO SETTLE
    # --------------------------------------------------------

    print(
        f"\nAllowing CARLA traffic "
        f"to settle for "
        f"{SETTLE_TIME:.1f}s..."
    )

    time.sleep(
        SETTLE_TIME
    )

    # --------------------------------------------------------
    # 6. SCENARIO FLAGS
    # --------------------------------------------------------

    (
        accident,
        road_blockage,
        emergency_vehicle_count
    ) = get_scenario_flags(
        scenario
    )

    # --------------------------------------------------------
    # 7. CREATE STATE ADAPTER
    # --------------------------------------------------------

    adapter = CarlaStateAdapter(
        world,
        scenario_tag=scenario.lower()
    )

    # --------------------------------------------------------
    # 8. READ STATE
    # --------------------------------------------------------

    state = adapter.get_state(
        timestep=0,
        accident=accident,
        road_blockage=road_blockage,
        emergency_vehicle_count=(
            emergency_vehicle_count
        )
    )

    print_state(
        state
    )

    # --------------------------------------------------------
    # 9. VERIFY CONTROLLED COUNT
    # --------------------------------------------------------

    if state.vehicle_count != len(
        vehicles
    ):

        print(
            "\nWARNING:"
        )

        print(
            "CARLA reports "
            f"{state.vehicle_count} "
            "vehicles while this "
            f"scenario spawned "
            f"{len(vehicles)}."
        )

        print(
            "Some external vehicle "
            "actors may still exist."
        )

    else:

        print(
            "\nVehicle count verification: PASS"
        )

    # --------------------------------------------------------
    # 10. CLEAN SCENARIO
    # --------------------------------------------------------

    print(
        "\nCleaning scenario traffic..."
    )

    for vehicle in vehicles:

        try:

            vehicle.set_autopilot(
                False
            )

            vehicle.destroy()

        except Exception:
            pass

    time.sleep(
        1.0
    )

    return state


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CITYTWIN AI — "
        "CARLA SCENARIO MANAGER"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        10.0
    )

    world = client.get_world()

    print(
        f"\nConnected to: "
        f"{world.get_map().name}"
    )

    # --------------------------------------------------------
    # SCENARIO MENU
    # --------------------------------------------------------

    print(
        "\nAvailable scenarios:"
    )

    for number, name in SCENARIOS.items():

        print(
            f"{number}. {name}"
        )

    print(
        "\nEnter 0 to run ALL scenarios."
    )

    choice = input(
        "\nSelect scenario: "
    ).strip()

    # --------------------------------------------------------
    # RESOLVE SELECTION
    # --------------------------------------------------------

    if choice == "0":

        selected_scenarios = list(
            SCENARIOS.values()
        )

    else:

        try:

            number = int(
                choice
            )

            if number not in SCENARIOS:
                raise ValueError

            selected_scenarios = [
                SCENARIOS[number]
            ]

        except ValueError:

            print(
                "\nInvalid scenario."
            )

            return

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    results = []

    try:

        for scenario in (
            selected_scenarios
        ):

            state = run_scenario(
                world,
                client,
                scenario
            )

            results.append(
                (
                    scenario,
                    state
                )
            )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print("\n\n")

        print("=" * 90)

        print(
            "SCENARIO SUMMARY"
        )

        print("=" * 90)

        print(
            f"\n{'Scenario':25s} "
            f"{'Vehicles':>9s} "
            f"{'Speed':>12s} "
            f"{'Density':>10s} "
            f"{'Queue':>12s} "
            f"{'Rain':>8s} "
            f"{'Fog':>8s}"
        )

        print(
            "-" * 90
        )

        for scenario, state in results:

            print(
                f"{scenario:25s} "
                f"{state.vehicle_count:9d} "
                f"{state.average_speed:12.2f} "
                f"{state.traffic_density:10.3f} "
                f"{state.queue_length:12.1f} "
                f"{state.rain_intensity:8.2f} "
                f"{state.fog_intensity:8.2f}"
            )

        print(
            "\nScenario Manager test complete."
        )

    finally:

        # ----------------------------------------------------
        # FINAL CLEANUP
        # ----------------------------------------------------

        cleanup_existing_traffic(
            world
        )

        # Reset weather.

        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        print(
            "CARLA weather reset."
        )

        print(
            "CARLA traffic cleaned."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()