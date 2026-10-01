import time
import random
import inspect

import carla
import torch

from carla_integration.state_adapter import CarlaStateAdapter
from prediction.predictor import TrafficPredictor
from risk.predictor import RiskPredictor
from rl.agent import DQNAgent
from rl.state_encoder import StateEncoder


NUM_VEHICLES = 10
SEQUENCE_LENGTH = 10
STEP_INTERVAL = 1.0

ACTION_NAMES = {
    0: "NO_ACTION",
    1: "EXTEND_GREEN",
    2: "REDUCE_GREEN",
    3: "TRAFFIC_DIVERSION",
    4: "EMERGENCY_PRIORITY",
    5: "COMBINED_RESPONSE",
}


def load_dqn():
    """
    Load the existing trained DQN without modifying the model.
    """

    dqn = DQNAgent(
        state_dim=20,
        action_dim=6
    )

    checkpoint = torch.load(
        "models/rl_agent.pt",
        map_location="cpu"
    )

    # Handle the checkpoint format used by the existing model.
    if isinstance(checkpoint, dict):

        if "policy_net_state_dict" in checkpoint:
            state_dict = checkpoint["policy_net_state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    dqn.policy_net.load_state_dict(state_dict)
    dqn.policy_net.eval()

    return dqn


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

    vehicle_blueprints = blueprint_library.filter(
        "vehicle.*"
    )

    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    spawned_vehicles = []

    for spawn_point in spawn_points:

        if len(spawned_vehicles) >= NUM_VEHICLES:
            break

        blueprint = random.choice(
            vehicle_blueprints
        )

        try:

            vehicle = world.try_spawn_actor(
                blueprint,
                spawn_point
            )

            if vehicle is not None:
                spawned_vehicles.append(vehicle)

        except Exception:
            pass

    print(
        f"Spawned vehicles: "
        f"{len(spawned_vehicles)}"
    )

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

    print(
        "Traffic Manager autopilot enabled."
    )

    # =========================================================
    # 4. LOAD LSTM
    # =========================================================

    print("\nLoading LSTM...")

    lstm_predictor = TrafficPredictor.from_checkpoint(
        "models/best_lstm_model.pt"
    )

    print("LSTM loaded.")

    # =========================================================
    # 5. LOAD RISK FNN
    # =========================================================

    print("\nLoading Risk FNN...")

    risk_predictor = RiskPredictor.from_checkpoint(
        "models/risk_model.pt"
    )

    print("Risk FNN loaded.")

    # =========================================================
    # 6. LOAD DQN
    # =========================================================

    print("\nLoading DQN...")

    dqn = load_dqn()

    print("DQN loaded.")

    # =========================================================
    # 7. CREATE STATE ADAPTER
    # =========================================================

    adapter = CarlaStateAdapter(
        world,
        scenario_tag="carla_dqn_test"
    )

    # =========================================================
    # 8. CREATE STATE ENCODER
    # =========================================================

    state_encoder = StateEncoder()

    print(
        "\nStateEncoder initialized."
    )

    print(
        f"StateEncoder.encode signature: "
        f"{inspect.signature(state_encoder.encode)}"
    )

    # =========================================================
    # 9. COLLECT 10 CARLA TIMESTEPS
    # =========================================================

    history = []

    print(
        f"\nCollecting {SEQUENCE_LENGTH} "
        "CARLA timesteps...\n"
    )

    for timestep in range(
        SEQUENCE_LENGTH
    ):

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
    # 10. LSTM FORECAST
    # =========================================================

    print("\n" + "=" * 70)
    print("STEP 1 — LSTM FORECAST")
    print("=" * 70)

    with torch.no_grad():

        forecast = lstm_predictor.predict(
            history
        )

    print("\nPredicted next 5 timesteps:\n")

    print(
        "Step | Traffic Density | "
        "Average Speed | Queue Length"
    )

    print("-" * 58)

    for i in range(5):

        density = forecast[
            "traffic_density"
        ][i]

        speed = forecast[
            "average_speed"
        ][i]

        queue = forecast[
            "queue_length"
        ][i]

        print(
            f" {i + 1:02d}  | "
            f"{density:15.4f} | "
            f"{speed:12.2f} km/h | "
            f"{queue:10.2f} m"
        )

    # =========================================================
    # 11. RISK PREDICTION
    # =========================================================

    print("\n" + "=" * 70)
    print("STEP 2 — FNN RISK ASSESSMENT")
    print("=" * 70)

    latest_state = history[-1]

    with torch.no_grad():

        risk_result = risk_predictor.predict(
            latest_state,
            forecast
        )

    risk_score = float(
        risk_result["risk_score"]
    )

    risk_level = risk_result[
        "risk_level"
    ]

    print(
        f"\nRisk Score : "
        f"{risk_score:.2f}/100"
    )

    print(
        f"Risk Level : "
        f"{risk_level}"
    )

    # =========================================================
    # 12. ENCODE STATE FOR DQN
    # =========================================================

    print("\n" + "=" * 70)
    print("STEP 3 — DQN STATE ENCODING")
    print("=" * 70)

    print(
        "\nCalling existing "
        "StateEncoder.encode()..."
    )

    # We deliberately inspect the actual encoder
    # signature rather than assuming its API.
    #
    # The encoder implementation determines exactly
    # how the 20-dimensional RL state is constructed.

    try:

        rl_state = state_encoder.encode(
            current_state=latest_state,
            forecast=forecast,
            risk_score=risk_score
        )

    except TypeError as error:

        print(
            "\nStateEncoder API mismatch:"
        )

        print(error)

        print(
            "\nThe existing encoder does not "
            "accept these keyword arguments."
        )

        print(
            "\nNo files were modified."
        )

        print(
            "Please send me the complete error "
            "above so we can match the existing "
            "encoder API exactly."
        )

        return

    # =========================================================
    # 13. CONVERT STATE TO TENSOR
    # =========================================================

    if hasattr(rl_state, "numpy"):

        rl_state_array = rl_state.numpy()

    else:

        rl_state_array = rl_state

    print(
        f"\nEncoded state type: "
        f"{type(rl_state_array)}"
    )

    print(
        f"Encoded state shape: "
        f"{getattr(rl_state_array, 'shape', None)}"
    )

    # =========================================================
    # 14. DQN DECISION
    # =========================================================

    print("\n" + "=" * 70)
    print("STEP 4 — DQN DECISION")
    print("=" * 70)

    state_tensor = torch.as_tensor(
        rl_state_array,
        dtype=torch.float32
    )

    if state_tensor.ndim == 1:

        state_tensor = state_tensor.unsqueeze(0)

    print(
        f"DQN input shape: "
        f"{tuple(state_tensor.shape)}"
    )

    with torch.no_grad():

        q_values = dqn.policy_net(
            state_tensor
        )

    action = int(
        torch.argmax(
            q_values,
            dim=1
        ).item()
    )

    print("\nQ-values:")

    for action_id in range(6):

        value = q_values[
            0,
            action_id
        ].item()

        print(
            f"  {action_id} "
            f"{ACTION_NAMES[action_id]:<20} "
            f"{value:8.4f}"
        )

    print(
        "\nSelected action:"
    )

    print(
        f"  {action} — "
        f"{ACTION_NAMES[action]}"
    )

    # =========================================================
    # 15. CLEANUP
    # =========================================================

    print("\nCleaning up...")

    for vehicle in spawned_vehicles:

        if vehicle.is_alive:

            vehicle.set_autopilot(
                False
            )

            vehicle.destroy()

    print("Cleanup complete.")


if __name__ == "__main__":
    main()