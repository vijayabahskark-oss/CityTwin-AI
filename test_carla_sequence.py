import time
import random

import carla

from carla_integration.state_adapter import CarlaStateAdapter


NUM_VEHICLES = 10
NUM_STEPS = 10
STEP_INTERVAL = 1.0


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(False)

    print(f"Connected to: {world.get_map().name}")

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    spawned_vehicles = []

    # Spawn vehicles
    for spawn_point in spawn_points:
        if len(spawned_vehicles) >= NUM_VEHICLES:
            break

        blueprint = random.choice(vehicle_blueprints)

        try:
            vehicle = world.try_spawn_actor(blueprint, spawn_point)

            if vehicle is not None:
                spawned_vehicles.append(vehicle)

        except Exception:
            pass

    print(f"Spawned vehicles: {len(spawned_vehicles)}")

    if not spawned_vehicles:
        print("No vehicles were spawned.")
        return

    # Enable Traffic Manager autopilot
    for vehicle in spawned_vehicles:
        vehicle.set_autopilot(True, traffic_manager.get_port())

    print("Traffic Manager autopilot enabled.")
    print(f"Collecting {NUM_STEPS} CityState timesteps...\n")

    adapter = CarlaStateAdapter(
        world,
        scenario_tag="carla_sequence_test",
    )

    history = []

    for timestep in range(NUM_STEPS):
        time.sleep(STEP_INTERVAL)

        state = adapter.get_state(timestep=timestep)
        history.append(state)

        print(
            f"[t={state.timestep:02d}] "
            f"vehicles={state.vehicle_count:3d} | "
            f"speed={state.average_speed:6.2f} km/h | "
            f"density={state.traffic_density:.3f} | "
            f"queue={state.queue_length:7.1f} m | "
            f"peds={state.pedestrian_count:3d} | "
            f"rain={state.rain_intensity:.2f} | "
            f"fog={state.fog_intensity:.2f}"
        )

    print("\n" + "=" * 70)
    print("CARLA SEQUENCE COLLECTION COMPLETE")
    print("=" * 70)

    print(f"\nCollected states: {len(history)}")

    print("\nLSTM input dimensions:")
    print(f"  Timesteps : {len(history)}")
    print("  Features  : 14")
    print("  Shape     : (1, 10, 14)")

    print("\nFirst state:")
    print(history[0].to_dict())

    print("\nLast state:")
    print(history[-1].to_dict())

    # Cleanup
    print("\nCleaning up...")

    for vehicle in spawned_vehicles:
        if vehicle.is_alive:
            vehicle.set_autopilot(False)
            vehicle.destroy()

    print("Cleanup complete.")


if __name__ == "__main__":
    main()