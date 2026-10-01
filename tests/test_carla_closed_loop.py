import sys
import time
import random
import numpy as np
import torch
import carla

sys.path.insert(0, ".")

from carla_integration.state_adapter import CarlaStateAdapter
from carla_integration.action_controller import CarlaActionController

from prediction.predictor import TrafficPredictor
from risk.predictor import RiskPredictor
from rl.agent import DQNAgent
from rl.state_encoder import StateEncoder


# ============================================================
# CONFIGURATION
# ============================================================

NUM_VEHICLES = 10
SEQUENCE_LENGTH = 10
DECISION_CYCLES = 5

OBSERVATION_INTERVAL = 0.5
ACTION_WAIT = 3.0

ACTION_NAMES = {
    0: "NO_ACTION",
    1: "EXTEND_GREEN",
    2: "REDUCE_GREEN",
    3: "TRAFFIC_DIVERSION",
    4: "EMERGENCY_PRIORITY",
    5: "COMBINED_RESPONSE",
}


# ============================================================
# SPAWN VEHICLES
# ============================================================

def spawn_vehicles(world, number):

    blueprint_library = world.get_blueprint_library()

    vehicle_blueprints = blueprint_library.filter(
        "vehicle.*"
    )

    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    vehicles = []

    for spawn_point in spawn_points:

        if len(vehicles) >= number:
            break

        blueprint = random.choice(
            vehicle_blueprints
        )

        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute(
                "role_name",
                "citytwin"
            )

        vehicle = world.try_spawn_actor(
            blueprint,
            spawn_point
        )

        if vehicle is not None:
            vehicles.append(vehicle)

    return vehicles


# ============================================================
# ENABLE AUTOPILOT
# ============================================================

def enable_autopilot(vehicles, client):

    # IMPORTANT:
    # Traffic Manager belongs to the CARLA Client,
    # not the World object.

    traffic_manager = client.get_trafficmanager(
        8000
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
# LOAD MODELS
# ============================================================

def load_models():

    print("\nLoading LSTM...")

    lstm = TrafficPredictor.from_checkpoint(
        "models/best_lstm_model.pt"
    )

    print("LSTM loaded.")

    print("\nLoading Risk FNN...")

    risk = RiskPredictor.from_checkpoint(
        "models/risk_model.pt"
    )

    print("Risk FNN loaded.")

    print("\nLoading DQN...")

    dqn = DQNAgent(
        state_dim=20,
        action_dim=6
    )

    checkpoint = torch.load(
        "models/rl_agent.pt",
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "policy_net_state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint[
                    "policy_net_state_dict"
                ]
            )

        elif "model_state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            dqn.policy_net.load_state_dict(
                checkpoint
            )

    else:

        dqn.policy_net.load_state_dict(
            checkpoint
        )

    dqn.policy_net.eval()

    print("DQN loaded.")

    return lstm, risk, dqn


# ============================================================
# COLLECT CARLA HISTORY
# ============================================================

def collect_history(
    adapter,
    sequence_length
):

    history = []

    print(
        f"\nCollecting "
        f"{sequence_length} CARLA observations..."
    )

    for t in range(sequence_length):

        state = adapter.get_state(
            timestep=t
        )

        history.append(state)

        print(
            f"[t={t:02d}] "
            f"speed={state.average_speed:6.2f} km/h | "
            f"density={state.traffic_density:.3f} | "
            f"queue={state.queue_length:7.1f} m"
        )

        time.sleep(
            OBSERVATION_INTERVAL
        )

    return history


# ============================================================
# LSTM FORECAST
# ============================================================

def run_forecast(
    lstm,
    history
):

    forecast = lstm.predict(
        history
    )

    print("\n" + "=" * 70)
    print("LSTM FORECAST")
    print("=" * 70)

    print(
        "\nStep | Traffic Density | Average Speed | Queue Length"
    )

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
            f"{i + 1:04d} | "
            f"{density:16.4f} | "
            f"{speed:13.2f} km/h | "
            f"{queue:11.2f} m"
        )

    return forecast


# ============================================================
# RISK ASSESSMENT
# ============================================================

def run_risk(
    risk_predictor,
    latest_state,
    forecast
):

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

    print("\n" + "=" * 70)
    print("RISK ASSESSMENT")
    print("=" * 70)

    print(
        f"\nRisk Score : "
        f"{risk_score:.2f}/100"
    )

    print(
        f"Risk Level : "
        f"{risk_level}"
    )

    return risk_score, risk_level


# ============================================================
# DQN DECISION
# ============================================================

def run_dqn(
    dqn,
    state_encoder,
    latest_state,
    forecast,
    risk_score
):

    rl_state = state_encoder.encode(
        current_state=latest_state,
        forecast=forecast,
        risk_score=risk_score
    )

    state_tensor = torch.tensor(
        rl_state,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = dqn.policy_net(
            state_tensor
        )

    q_values = (
        q_values
        .cpu()
        .numpy()[0]
    )

    action = int(
        np.argmax(q_values)
    )

    print("\n" + "=" * 70)
    print("DQN DECISION")
    print("=" * 70)

    print(
        f"\nDQN state shape: "
        f"{rl_state.shape}"
    )

    print("\nQ-values:")

    for i, q in enumerate(q_values):

        print(
            f"{i} "
            f"{ACTION_NAMES[i]:22s} "
            f"{q:10.4f}"
        )

    print(
        f"\nSelected action:\n"
        f"{action} — "
        f"{ACTION_NAMES[action]}"
    )

    return action


# ============================================================
# POST-ACTION STATE
# ============================================================

def print_post_action_state(
    adapter,
    action,
    risk_score,
    risk_level
):

    state = adapter.get_state(
        timestep=SEQUENCE_LENGTH
    )

    print("\n" + "=" * 70)
    print("POST-ACTION CARLA STATE")
    print("=" * 70)

    print(
        f"\nAction          : "
        f"{ACTION_NAMES[action]}"
    )

    print(
        f"Vehicle count   : "
        f"{state.vehicle_count}"
    )

    print(
        f"Average speed   : "
        f"{state.average_speed:.2f} km/h"
    )

    print(
        f"Traffic density : "
        f"{state.traffic_density:.3f}"
    )

    print(
        f"Queue length    : "
        f"{state.queue_length:.2f} m"
    )

    print(
        f"Risk before     : "
        f"{risk_score:.2f} "
        f"({risk_level})"
    )

    return state


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CITYTWIN AI — CLOSED LOOP CARLA PROTOTYPE")
    print("=" * 70)

    # --------------------------------------------------------
    # CONNECT TO CARLA
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
    # SPAWN TRAFFIC
    # --------------------------------------------------------

    vehicles = spawn_vehicles(
        world,
        NUM_VEHICLES
    )

    print(
        f"Spawned vehicles: "
        f"{len(vehicles)}"
    )

    enable_autopilot(
        vehicles,
        client
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    lstm, risk_predictor, dqn = (
        load_models()
    )

    state_encoder = StateEncoder()

    print(
        "\nStateEncoder initialized."
    )

    # --------------------------------------------------------
    # CARLA INTEGRATION
    # --------------------------------------------------------

    adapter = CarlaStateAdapter(
        world,
        scenario_tag="closed_loop"
    )

    action_controller = (
        CarlaActionController(
            world
        )
    )

    # --------------------------------------------------------
    # CLOSED LOOP
    # --------------------------------------------------------

    try:

        for cycle in range(
            DECISION_CYCLES
        ):

            print("\n\n")
            print(
                "#" * 70
            )

            print(
                f"DECISION CYCLE "
                f"{cycle + 1}/"
                f"{DECISION_CYCLES}"
            )

            print(
                "#" * 70
            )

            # ------------------------------------------------
            # STEP 1 — OBSERVE
            # ------------------------------------------------

            history = collect_history(
                adapter,
                SEQUENCE_LENGTH
            )

            latest_state = (
                history[-1]
            )

            # ------------------------------------------------
            # STEP 2 — PREDICT
            # ------------------------------------------------

            forecast = run_forecast(
                lstm,
                history
            )

            # ------------------------------------------------
            # STEP 3 — RISK
            # ------------------------------------------------

            (
                risk_score,
                risk_level
            ) = run_risk(
                risk_predictor,
                latest_state,
                forecast
            )

            # ------------------------------------------------
            # STEP 4 — DQN
            # ------------------------------------------------

            action = run_dqn(
                dqn,
                state_encoder,
                latest_state,
                forecast,
                risk_score
            )

            # ------------------------------------------------
            # STEP 5 — ACT
            # ------------------------------------------------

            print("\n" + "=" * 70)
            print("APPLYING DQN ACTION")
            print("=" * 70)

            action_result = (
                action_controller.apply_action(
                    action
                )
            )

            print(
                f"\nAction applied: "
                f"{action_result}"
            )

            # ------------------------------------------------
            # STEP 6 — WAIT
            # ------------------------------------------------

            print(
                f"\nWaiting "
                f"{ACTION_WAIT:.1f}s "
                f"for CARLA response..."
            )

            time.sleep(
                ACTION_WAIT
            )

            # ------------------------------------------------
            # STEP 7 — OBSERVE AGAIN
            # ------------------------------------------------

            post_state = (
                print_post_action_state(
                    adapter,
                    action,
                    risk_score,
                    risk_level
                )
            )

            # ------------------------------------------------
            # CYCLE SUMMARY
            # ------------------------------------------------

            print("\n" + "-" * 70)
            print("CYCLE SUMMARY")
            print("-" * 70)

            print(
                f"Cycle           : "
                f"{cycle + 1}"
            )

            print(
                f"Risk            : "
                f"{risk_score:.2f} "
                f"({risk_level})"
            )

            print(
                f"DQN action      : "
                f"{ACTION_NAMES[action]}"
            )

            print(
                f"Post speed      : "
                f"{post_state.average_speed:.2f} km/h"
            )

            print(
                f"Post density    : "
                f"{post_state.traffic_density:.3f}"
            )

            print(
                f"Post queue      : "
                f"{post_state.queue_length:.2f} m"
            )

            print(
                "-" * 70
            )

            print(
                "\nCycle complete."
            )

    finally:

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        print("\n")
        print(
            "=" * 70
        )

        print(
            "CLEANING UP CARLA ACTORS"
        )

        print(
            "=" * 70
        )

        for vehicle in vehicles:

            try:
                vehicle.destroy()

            except Exception:
                pass

        print(
            "Cleanup complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()