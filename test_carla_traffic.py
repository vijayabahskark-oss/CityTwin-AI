import time
import random

import carla

from carla_integration.state_adapter import CarlaStateAdapter


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print(f"Connected to: {world.get_map().name}")

    # Get vehicle blueprints
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    spawned_vehicles = []

    # Spawn up to 10 vehicles
    for spawn_point in spawn_points:
        if len(spawned_vehicles) >= 10:
            break

        blueprint = random.choice(vehicle_blueprints)

        try:
            vehicle = world.try_spawn_actor(blueprint, spawn_point)

            if vehicle is not None:
                spawned_vehicles.append(vehicle)

        except Exception:
            pass

    print(f"Spawned vehicles: {len(spawned_vehicles)}")

    # Give CARLA some time to register the actors
    time.sleep(3)

    # Read CARLA state
    adapter = CarlaStateAdapter(
        world,
        scenario_tag="carla_traffic_test",
    )

    state = adapter.get_state(timestep=0)

    print("\n## CARLA → CityState\n")
    print(
        f"[t={state.timestep:04d} | "
        f"{state.scenario_tag:<20}] "
        f"vehicles={state.vehicle_count:3d}  "
        f"speed={state.average_speed:5.1f}km/h  "
        f"density={state.traffic_density:.3f}  "
        f"queue={state.queue_length:6.1f}m  "
        f"peds={state.pedestrian_count:3d}  "
        f"rain={state.rain_intensity:.2f}  "
        f"fog={state.fog_intensity:.2f}  "
        f"signal={state.traffic_signal_state:<8} "
        f"incidents="
        f"{'accident' if state.accident else 'none'}"
    )

    print("\nDictionary:")
    print(state.to_dict())

    # Cleanup
    print("\nCleaning up spawned vehicles...")

    for vehicle in spawned_vehicles:
        if vehicle.is_alive:
            vehicle.destroy()

    print("Cleanup complete.")


if __name__ == "__main__":
    main()