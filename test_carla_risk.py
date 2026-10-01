import time
import random

import carla
import torch

from carla_integration.state_adapter import CarlaStateAdapter
from prediction.predictor import TrafficPredictor
from risk.predictor import RiskPredictor


NUM_VEHICLES = 10
SEQUENCE_LENGTH = 10
STEP_INTERVAL = 1.0


def main():

    # =========================================================
    # 1. CONNECT TO CARLA
    # =========================================================
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)

    print(f"Connected to: {world.get_map().name}")

    # =========================================================
    # 2. SPAWN VEHICLES
    # =========================================================
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    spawned_vehicles = []

    for spawn_point in spawn_points:

        if len(spawned_vehicles) >= NUM_VEHICLES:
            break

        blueprint = random.choice(vehicle_blueprints)

        try:
            vehicle = world.try_spawn_actor(
                blueprint,
                spawn_point
            )

            if vehicle is not None:
                spawned_vehicles.append(vehicle)

        except Exception:
            pass

    print(f"Spawned vehicles: {len(spawned_vehicles)}")

    if not spawned_vehicles:
        print("ERROR: No vehicles spawned.")
        return

    # =========================================================
    # 3. ENABLE TRAFFIC MANAGER
    # =========================================================
    for vehicle in spawned_vehicles:

        vehicle.set_autopilot(
            True,
            traffic_manager.get_port()
        )

    print("Traffic Manager autopilot enabled.")

    # =========================================================
    # 4. LOAD TRAINED MODELS
    # =========================================================
    print("\nLoading trained LSTM...")

    lstm_predictor = TrafficPredictor.from_checkpoint(
        "models/best_lstm_model.pt"
    )

    print("LSTM loaded successfully.")

    print("\nLoading trained Risk FNN...")

    risk_predictor = RiskPredictor.from_checkpoint(
        "models/risk_model.pt"
    )

    print("Risk FNN loaded successfully.")

    # =========================================================
    # 5. CREATE CARLA STATE ADAPTER
    # =========================================================
    adapter = CarlaStateAdapter(
        world,
        scenario_tag="carla_risk_test"
    )

    # =========================================================
    # 6. COLLECT 10 CARLA TIMESTEPS
    # =========================================================
    history = []

    print(
        f"\nCollecting {SEQUENCE_LENGTH} "
        "CARLA timesteps...\n"
    )

    for timestep in range(SEQUENCE_LENGTH):

        time.sleep(STEP_INTERVAL)

        state = adapter.get_state(
            timestep=timestep
        )

        history.append(state)

        print(
            f"[t={timestep:02d}] "
            f"speed={state.average_speed:6.2f} km/h | "
            f"density={state.traffic_density:.3f} | "
            f"queue={state.queue_length:7.1f} m"
        )

    # =========================================================
    # 7. LSTM — PREDICT FUTURE TRAFFIC
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 1 — LSTM TRAFFIC FORECAST")
    print("=" * 70)

    print("\nInput:")
    print("  Timesteps : 10")
    print("  Features  : 14")
    print("  Shape     : (1, 10, 14)")

    with torch.no_grad():

        forecast = lstm_predictor.predict(history)

    print("\nPredicted next 5 timesteps:\n")

    print(
        "Step | Traffic Density | Average Speed | Queue Length"
    )

    print("-" * 58)

    for i in range(5):

        density = forecast["traffic_density"][i]
        speed = forecast["average_speed"][i]
        queue = forecast["queue_length"][i]

        print(
            f" {i + 1:02d}  | "
            f"{density:15.4f} | "
            f"{speed:12.2f} km/h | "
            f"{queue:10.2f} m"
        )

    # =========================================================
    # 8. FNN — CALCULATE URBAN DISRUPTION RISK
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 2 — FNN URBAN DISRUPTION RISK")
    print("=" * 70)

    # Risk prediction uses the latest observed CARLA state
    # together with the LSTM forecast.
    latest_state = history[-1]

    with torch.no_grad():

        risk_result = risk_predictor.predict(
            latest_state,
            forecast
        )

    print("\nRisk prediction:")

    print(risk_result)

    # =========================================================
    # 9. DISPLAY RISK RESULT
    # =========================================================
    if isinstance(risk_result, dict):

        risk_score = risk_result.get(
            "risk_score",
            risk_result.get("score")
        )

        risk_level = risk_result.get(
            "risk_level",
            risk_result.get("level")
        )

        if risk_score is not None:

            print(
                f"\nUrban Disruption Risk Score: "
                f"{float(risk_score):.2f}/100"
            )

        if risk_level is not None:

            print(
                f"Risk Level: {risk_level}"
            )

    else:

        print(
            f"\nRisk result: {risk_result}"
        )

    # =========================================================
    # 10. CLEANUP
    # =========================================================
    print("\nCleaning up...")

    for vehicle in spawned_vehicles:

        if vehicle.is_alive:

            vehicle.set_autopilot(False)
            vehicle.destroy()

    print("Cleanup complete.")


if __name__ == "__main__":
    main()