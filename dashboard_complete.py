import time
import random
import threading
import inspect
from pathlib import Path

import torch

import carla
import numpy as np
import streamlit as st

from perception.predictor import UrbanPerceptionPredictor
from prediction.predictor import TrafficPredictor
from risk.predictor import RiskPredictor
from carla_integration.state_adapter import CarlaStateAdapter
from rl.state_encoder import StateEncoder
from rl.agent import DQNAgent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CityTwin AI",
    page_icon="🏙️",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT = 8000

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FOV = 90

ROOT_DIR = Path(__file__).resolve().parent
CNN_CHECKPOINT = ROOT_DIR / "models" / "urban_cnn.pt"
LSTM_CHECKPOINT = ROOT_DIR / "models" / "best_lstm_model.pt"
RISK_CHECKPOINT = ROOT_DIR / "models" / "risk_model.pt"
DQN_CHECKPOINT_CANDIDATES = [
    ROOT_DIR / "models" / "rl_agent.pt",
    ROOT_DIR / "models" / "best_rl_agent.pt",
    ROOT_DIR / "models" / "rl_agent.pt",
    ROOT_DIR / "models" / "rl_agent.pt",
]


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = {
    "NORMAL": {
        "vehicles": 10,
        "rain": 0.0,
        "fog": 0.0,
        "cloudiness": 0.0,
        "wetness": 0.0,
        "wind": 0.0,
        "accident": False,
        "road_blockage": False,
        "emergency": 0,
    },

    "RUSH_HOUR": {
        "vehicles": 30,
        "rain": 0.0,
        "fog": 0.0,
        "cloudiness": 0.0,
        "wetness": 0.0,
        "wind": 0.0,
        "accident": False,
        "road_blockage": False,
        "emergency": 0,
    },

    "HEAVY_RAIN": {
        "vehicles": 15,
        "rain": 90.0,
        "fog": 15.0,
        "cloudiness": 80.0,
        "wetness": 90.0,
        "wind": 40.0,
        "accident": False,
        "road_blockage": False,
        "emergency": 0,
    },

    "FOG": {
        "vehicles": 15,
        "rain": 0.0,
        "fog": 85.0,
        "cloudiness": 70.0,
        "wetness": 20.0,
        "wind": 0.0,
        "accident": False,
        "road_blockage": False,
        "emergency": 0,
    },

    "ACCIDENT": {
        "vehicles": 15,
        "rain": 0.0,
        "fog": 0.0,
        "cloudiness": 0.0,
        "wetness": 0.0,
        "wind": 0.0,
        "accident": True,
        "road_blockage": False,
        "emergency": 0,
    },

    "ROAD_BLOCKAGE": {
        "vehicles": 15,
        "rain": 0.0,
        "fog": 0.0,
        "cloudiness": 0.0,
        "wetness": 0.0,
        "wind": 0.0,
        "accident": False,
        "road_blockage": True,
        "emergency": 0,
    },

    "EMERGENCY": {
        "vehicles": 15,
        "rain": 0.0,
        "fog": 0.0,
        "cloudiness": 0.0,
        "wetness": 0.0,
        "wind": 0.0,
        "accident": False,
        "road_blockage": False,
        "emergency": 1,
    },

    "HEAVY_RAIN_ACCIDENT": {
        "vehicles": 20,
        "rain": 95.0,
        "fog": 30.0,
        "cloudiness": 90.0,
        "wetness": 95.0,
        "wind": 50.0,
        "accident": True,
        "road_blockage": False,
        "emergency": 0,
    },
}


# ============================================================
# CARLA CONNECTION
# ============================================================

@st.cache_resource
def connect_carla():

    client = carla.Client(
        CARLA_HOST,
        CARLA_PORT,
    )

    client.set_timeout(10.0)

    world = client.get_world()

    traffic_manager = client.get_trafficmanager(
        TM_PORT
    )

    traffic_manager.set_synchronous_mode(False)

    return client, world, traffic_manager


try:

    client, world, traffic_manager = connect_carla()

except Exception as e:

    st.error(
        "Could not connect to CARLA. "
        "Make sure CarlaUE4.exe is running."
    )

    st.exception(e)
    st.stop()


# ============================================================
# CNN MODEL
# ============================================================

@st.cache_resource
def load_cnn_predictor():

    if not CNN_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Urban CNN checkpoint not found:\n"
            f"{CNN_CHECKPOINT}"
        )

    predictor = UrbanPerceptionPredictor.from_checkpoint(
        CNN_CHECKPOINT,
        device="cpu",
    )

    return predictor


try:

    cnn_predictor = load_cnn_predictor()

    cnn_status = True

except Exception as e:

    cnn_predictor = None
    cnn_status = False
    cnn_error = str(e)


# ============================================================
# LSTM TRAFFIC FORECASTER
# ============================================================

@st.cache_resource
def load_lstm_predictor():

    if not LSTM_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Traffic LSTM checkpoint not found:\n"
            f"{LSTM_CHECKPOINT}"
        )

    return TrafficPredictor.from_checkpoint(
        LSTM_CHECKPOINT,
        device="cpu",
    )


try:

    lstm_predictor = load_lstm_predictor()
    lstm_status = True

except Exception as e:

    lstm_predictor = None
    lstm_status = False
    lstm_error = str(e)


# ============================================================
# RISK FNN PREDICTOR
# ============================================================

@st.cache_resource
def load_risk_predictor():

    if not RISK_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Risk model checkpoint not found:\n"
            f"{RISK_CHECKPOINT}"
        )

    return RiskPredictor.from_checkpoint(
        RISK_CHECKPOINT,
        device="cpu",
    )


try:

    risk_predictor = load_risk_predictor()
    risk_status = True

except Exception as e:

    risk_predictor = None
    risk_status = False
    risk_error = str(e)



# ============================================================
# DQN CONTROLLER
# ============================================================

ACTION_NAMES = {
    0: "NO_ACTION",
    1: "EXTEND_GREEN",
    2: "REDUCE_GREEN",
    3: "TRAFFIC_DIVERSION",
    4: "EMERGENCY_PRIORITY",
    5: "COMBINED_RESPONSE",
}

# This is the exact checkpoint used by the working
# tests/test_carla_closed_loop.py script.
DQN_CHECKPOINT = ROOT_DIR / "models" / "rl_agent.pt"


@st.cache_resource
def load_dqn_controller():
    """
    Load the trained DQN exactly as the project's closed-loop
    CARLA test does.

    Architecture:
        state_dim = 20
        action_dim = 6

    Network used for inference:
        dqn.policy_net
    """

    if not DQN_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {DQN_CHECKPOINT}"
        )

    dqn = DQNAgent(
        state_dim=20,
        action_dim=6,
    )

    checkpoint = torch.load(
        DQN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "policy_net_state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint["policy_net_state_dict"]
            )

        elif "model_state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            dqn.policy_net.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            # Support a raw PyTorch state_dict stored directly
            # as the checkpoint dictionary.
            try:
                dqn.policy_net.load_state_dict(checkpoint)
            except Exception as exc:
                raise RuntimeError(
                    "rl_agent.pt is a dictionary, but it does not "
                    "contain a compatible DQN policy network state_dict."
                ) from exc

    else:
        # Support a raw state_dict object.
        dqn.policy_net.load_state_dict(checkpoint)

    dqn.policy_net.eval()

    return dqn


try:

    dqn_agent = load_dqn_controller()

    dqn_status = True
    dqn_error = ""

except Exception as exc:

    dqn_agent = None
    dqn_status = False
    dqn_error = str(exc)


@st.cache_resource
def get_state_encoder():
    return StateEncoder()


class DashboardActionController:
    """
    Dashboard-side CARLA action executor.

    The project module carla_integration.action_controller.py does
    not expose an ActionController class, so the dashboard keeps
    this small executor local instead of importing a nonexistent
    class.

    The six DQN actions are mapped to the first available CARLA
    traffic light, matching the project's MVP closed-loop behavior.
    """

    def __init__(self, carla_world):
        self.world = carla_world
        self._refresh_traffic_lights()

    def _refresh_traffic_lights(self):
        self.traffic_lights = list(
            self.world.get_actors().filter(
                "traffic.traffic_light"
            )
        )

    def _get_light(self):
        if not self.traffic_lights:
            self._refresh_traffic_lights()

        if not self.traffic_lights:
            raise RuntimeError(
                "No CARLA traffic lights are available."
            )

        return self.traffic_lights[0]

    def apply_action(self, action):
        action = int(action)

        if action not in ACTION_NAMES:
            raise ValueError(
                f"Unsupported DQN action: {action}"
            )

        light = self._get_light()

        if action == 0:
            return {
                "action": action,
                "action_name": ACTION_NAMES[action],
                "applied": True,
                "traffic_light_id": light.id,
                "traffic_light_state": str(
                    light.get_state()
                ),
            }

        # Traffic-light MVP mapping:
        # 1 EXTEND_GREEN          -> Green
        # 2 REDUCE_GREEN          -> Red
        # 3 TRAFFIC_DIVERSION     -> Red
        # 4 EMERGENCY_PRIORITY    -> Green
        # 5 COMBINED_RESPONSE     -> Green
        if action in (1, 4, 5):

            light.set_state(
                carla.TrafficLightState.Green
            )

            state_name = "Green"

        else:

            light.set_state(
                carla.TrafficLightState.Red
            )

            state_name = "Red"

        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "applied": True,
            "traffic_light_id": light.id,
            "traffic_light_state": state_name,
        }


@st.cache_resource
def get_action_controller():
    return DashboardActionController(world)


def _dqn_q_values(agent, state):
    """
    Exact inference path used by the working closed-loop test:
        dqn.policy_net(state_tensor)
    """

    state_np = np.asarray(
        state,
        dtype=np.float32,
    )

    if state_np.shape != (20,):
        raise ValueError(
            f"DQN expects a 20-dimensional state, "
            f"got shape {state_np.shape}"
        )

    state_tensor = torch.from_numpy(
        state_np
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

    return (
        q_values
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


def choose_dqn_action(agent, state):

    q_values = _dqn_q_values(
        agent,
        state,
    )

    return int(
        np.argmax(q_values)
    )


def apply_dqn_action(action):

    controller = get_action_controller()

    return controller.apply_action(
        int(action)
    )


# ============================================================
# CAMERA BUFFER
# ============================================================

class CameraBuffer:

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_frame = None

        self.frame_number = 0

    def push(self, image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8,
        )

        array = np.reshape(
            array,
            (
                image.height,
                image.width,
                4,
            ),
        )

        # CARLA raw format:
        # BGRA -> RGB
        rgb = array[:, :, :3][:, :, ::-1].copy()

        with self.lock:

            self.latest_frame = rgb

            self.frame_number += 1

    def get(self):

        with self.lock:

            return (
                self.latest_frame,
                self.frame_number,
            )

    def clear(self):

        with self.lock:

            self.latest_frame = None

            self.frame_number = 0


@st.cache_resource
def get_camera_buffer():

    return CameraBuffer()


camera_buffer = get_camera_buffer()


# ============================================================
# SESSION STATE
# ============================================================

if "scenario_running" not in st.session_state:

    st.session_state.scenario_running = False


if "active_scenario" not in st.session_state:

    st.session_state.active_scenario = None


if "scenario_actors" not in st.session_state:

    st.session_state.scenario_actors = []


if "scenario_flags" not in st.session_state:

    st.session_state.scenario_flags = {}


if "cycle_count" not in st.session_state:

    st.session_state.cycle_count = 0


if "camera_actor" not in st.session_state:

    st.session_state.camera_actor = None

if "lstm_forecast" not in st.session_state:

    st.session_state.lstm_forecast = None

if "lstm_history" not in st.session_state:

    st.session_state.lstm_history = []

if "lstm_message" not in st.session_state:

    st.session_state.lstm_message = ""

if "risk_result" not in st.session_state:

    st.session_state.risk_result = None

if "risk_message" not in st.session_state:

    st.session_state.risk_message = ""

if "lstm_current_state" not in st.session_state:

    st.session_state.lstm_current_state = None

if "dqn_result" not in st.session_state:

    st.session_state.dqn_result = None

if "dqn_message" not in st.session_state:

    st.session_state.dqn_message = ""


# ============================================================
# CAMERA CALLBACK
# ============================================================

def camera_callback(image):

    camera_buffer.push(image)


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    camera = st.session_state.get(
        "camera_actor"
    )

    if camera is not None:

        try:

            camera.stop()

        except Exception:

            pass

        try:

            camera.destroy()

        except Exception:

            pass

    st.session_state.camera_actor = None

    camera_buffer.clear()


# ============================================================
# START CAMERA
# ============================================================

def start_camera(vehicle):

    stop_camera()

    blueprint_library = (
        world.get_blueprint_library()
    )

    camera_bp = blueprint_library.find(
        "sensor.camera.rgb"
    )

    camera_bp.set_attribute(
        "image_size_x",
        str(CAMERA_WIDTH),
    )

    camera_bp.set_attribute(
        "image_size_y",
        str(CAMERA_HEIGHT),
    )

    camera_bp.set_attribute(
        "fov",
        str(CAMERA_FOV),
    )

    camera_transform = carla.Transform(
        carla.Location(
            x=1.5,
            z=2.2,
        ),
        carla.Rotation(
            pitch=-5.0,
            yaw=0.0,
            roll=0.0,
        ),
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    camera.listen(
        camera_callback
    )

    st.session_state.camera_actor = camera


# ============================================================
# CLEAN TRAFFIC
# ============================================================

def cleanup_traffic():

    stop_camera()

    actors = world.get_actors().filter(
        "vehicle.*"
    )

    removed = 0

    for actor in actors:

        try:

            actor.destroy()

            removed += 1

        except Exception:

            pass

    st.session_state.scenario_actors = []

    return removed


# ============================================================
# APPLY WEATHER
# ============================================================

def apply_weather(config):

    weather = carla.WeatherParameters()

    weather.cloudiness = (
        config["cloudiness"]
    )

    weather.precipitation = (
        config["rain"]
    )

    weather.precipitation_deposits = (
        config["wetness"]
    )

    weather.wind_intensity = (
        config["wind"]
    )

    weather.fog_density = (
        config["fog"]
    )

    world.set_weather(weather)


# ============================================================
# SPAWN VEHICLES
# ============================================================

def spawn_vehicles(count):

    blueprint_library = (
        world.get_blueprint_library()
    )

    vehicle_blueprints = list(
        blueprint_library.filter(
            "vehicle.*"
        )
    )

    spawn_points = list(
        world.get_map().get_spawn_points()
    )

    random.shuffle(spawn_points)

    spawned = []

    for spawn_point in spawn_points:

        if len(spawned) >= count:

            break

        if not vehicle_blueprints:

            break

        blueprint = random.choice(
            vehicle_blueprints
        )

        try:

            vehicle = world.try_spawn_actor(
                blueprint,
                spawn_point,
            )

            if vehicle is not None:

                spawned.append(vehicle)

        except Exception:

            continue

    return spawned


# ============================================================
# ENABLE AUTOPILOT
# ============================================================

def enable_autopilot(vehicles):

    for vehicle in vehicles:

        try:

            vehicle.set_autopilot(
                True,
                traffic_manager.get_port(),
            )

        except Exception:

            pass


# ============================================================
# START SCENARIO
# ============================================================

def start_scenario(name):

    config = SCENARIOS[name]

    cleanup_traffic()

    apply_weather(config)

    vehicles = spawn_vehicles(
        config["vehicles"]
    )

    enable_autopilot(vehicles)

    st.session_state.scenario_running = True

    st.session_state.active_scenario = name

    st.session_state.scenario_actors = vehicles

    st.session_state.scenario_flags = {
        "accident": config["accident"],
        "road_blockage": config["road_blockage"],
        "emergency": config["emergency"],
    }

    st.session_state.cycle_count = 0
    st.session_state.lstm_forecast = None
    st.session_state.lstm_history = []
    st.session_state.lstm_message = ""
    st.session_state.risk_result = None
    st.session_state.risk_message = ""
    st.session_state.lstm_current_state = None
    st.session_state.dqn_result = None
    st.session_state.dqn_message = ""

    # Attach camera to first vehicle.
    if vehicles:

        start_camera(
            vehicles[0]
        )

    return len(vehicles)


# ============================================================
# STOP SCENARIO
# ============================================================

def stop_scenario():

    cleanup_traffic()

    world.set_weather(
        carla.WeatherParameters.ClearNoon
    )

    st.session_state.scenario_running = False

    st.session_state.active_scenario = None

    st.session_state.scenario_flags = {}

    st.session_state.cycle_count = 0
    st.session_state.lstm_forecast = None
    st.session_state.lstm_history = []
    st.session_state.lstm_message = ""
    st.session_state.risk_result = None
    st.session_state.risk_message = ""
    st.session_state.lstm_current_state = None
    st.session_state.dqn_result = None
    st.session_state.dqn_message = ""


# ============================================================
# CURRENT CITY STATE
# ============================================================

def get_basic_state():

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    speeds = []

    for vehicle in vehicles:

        velocity = vehicle.get_velocity()

        speed_ms = (
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        ) ** 0.5

        speeds.append(
            speed_ms * 3.6
        )

    vehicle_count = len(vehicles)

    average_speed = (
        sum(speeds) / len(speeds)
        if speeds
        else 0.0
    )

    density = min(
        1.0,
        vehicle_count / 200.0,
    )

    queue_count = sum(
        1
        for speed in speeds
        if speed <= 5.0
    )

    queue_length = (
        queue_count * 7.0
    )

    weather = world.get_weather()

    rain = (
        weather.precipitation / 100.0
    )

    fog = (
        weather.fog_density / 100.0
    )

    return {
        "vehicles": vehicle_count,
        "speed": average_speed,
        "density": density,
        "queue": queue_length,
        "rain": rain,
        "fog": fog,
    }


# ============================================================
# LSTM OBSERVATION COLLECTION
# ============================================================

def collect_lstm_forecast():

    if not st.session_state.scenario_running:

        raise RuntimeError(
            "Start a CARLA scenario before running the LSTM forecast."
        )

    if not lstm_status or lstm_predictor is None:

        raise RuntimeError(
            "LSTM predictor is not available."
        )

    if not risk_status or risk_predictor is None:

        raise RuntimeError(
            "Risk FNN predictor is not available."
        )

    adapter = CarlaStateAdapter(
        world,
        scenario_tag=(
            f"carla_{st.session_state.active_scenario.lower()}"
        ),
    )

    sequence = []
    total_steps = lstm_predictor.seq_len
    flags = st.session_state.scenario_flags

    for step in range(total_steps):

        city_state = adapter.get_state(
            timestep=(
                st.session_state.cycle_count * total_steps
                + step
            ),
            accident=flags.get("accident", False),
            road_blockage=flags.get("road_blockage", False),
            emergency_vehicle_count=flags.get("emergency", 0),
        )

        sequence.append(city_state)

        if step < total_steps - 1:
            time.sleep(0.5)

    forecast = lstm_predictor.predict(sequence)

    # Risk assessment uses the latest observed CityState
    # together with the complete LSTM future trajectory.
    current_state = sequence[-1]

    risk_result = risk_predictor.predict(
        current_state,
        forecast,
    )

    return sequence, forecast, risk_result, current_state


# ============================================================
# HEADER
# ============================================================

st.title("🏙️ CityTwin AI")

st.caption(
    "Intelligent Urban Event Prediction & Adaptive Response"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎛️ Scenario Control"
)

scenario = st.sidebar.selectbox(
    "Select Scenario",
    list(SCENARIOS.keys()),
)

config = SCENARIOS[scenario]

st.sidebar.write(
    f"**Target vehicles:** "
    f"{config['vehicles']}"
)

st.sidebar.write(
    f"**Rain:** "
    f"{config['rain'] / 100:.2f}"
)

st.sidebar.write(
    f"**Fog:** "
    f"{config['fog'] / 100:.2f}"
)

if config["accident"]:

    st.sidebar.warning(
        "⚠ Accident condition"
    )

if config["road_blockage"]:

    st.sidebar.warning(
        "🚧 Road blockage condition"
    )

if config["emergency"]:

    st.sidebar.info(
        "🚑 Emergency vehicle condition"
    )

st.sidebar.divider()

start_button = st.sidebar.button(
    "▶ Start Scenario",
    use_container_width=True,
)

stop_button = st.sidebar.button(
    "■ Stop Scenario",
    use_container_width=True,
)

cycle_button = st.sidebar.button(
    "🔄 Run AI Cycle",
    use_container_width=True,
    disabled=not st.session_state.scenario_running,
)

lstm_button = st.sidebar.button(
    "🔮 Run LSTM Forecast",
    use_container_width=True,
    disabled=(
        not st.session_state.scenario_running
        or not lstm_status
        or not risk_status
    ),
)

dqn_button = st.sidebar.button(
    "🤖 Run DQN Decision",
    use_container_width=True,
    disabled=(
        not st.session_state.scenario_running
        or not dqn_status
        or st.session_state.lstm_forecast is None
        or st.session_state.risk_result is None
    ),
)

if not dqn_status:
    st.sidebar.warning("DQN controller unavailable: models/rl_agent.pt")

if not risk_status:

    st.sidebar.warning(
        "Risk FNN unavailable"
    )


# ============================================================
# BUTTON ACTIONS
# ============================================================

if start_button:

    with st.spinner(
        f"Starting {scenario}..."
    ):

        spawned = start_scenario(
            scenario
        )

        time.sleep(3)

    st.success(
        f"{scenario} started — "
        f"{spawned} vehicles spawned."
    )

    st.rerun()


if stop_button:

    stop_scenario()

    st.success(
        "Scenario stopped and CARLA reset."
    )

    st.rerun()


if cycle_button:

    st.session_state.cycle_count += 1

    st.toast(
        f"AI cycle "
        f"{st.session_state.cycle_count} "
        f"executed."
    )

    st.rerun()

if lstm_button:

    st.session_state.lstm_forecast = None
    st.session_state.lstm_history = []
    st.session_state.lstm_message = ""
    st.session_state.risk_result = None
    st.session_state.risk_message = ""
    st.session_state.lstm_current_state = None
    st.session_state.dqn_result = None
    st.session_state.dqn_message = ""

    with st.spinner(
        f"Collecting {lstm_predictor.seq_len} CARLA observations "
        "and generating the LSTM forecast..."
    ):

        try:

            history, forecast, risk_result, current_state = collect_lstm_forecast()

            st.session_state.lstm_history = history
            st.session_state.lstm_forecast = forecast
            st.session_state.risk_result = risk_result
            st.session_state.lstm_current_state = current_state
            st.session_state.dqn_result = None
            st.session_state.dqn_message = ""
            st.session_state.lstm_message = (
                f"Forecast generated from {len(history)} "
                "CARLA observations."
            )
            st.session_state.risk_message = (
                "Risk assessment generated from the latest "
                "CARLA state and LSTM forecast."
            )

        except Exception as e:

            st.session_state.lstm_forecast = None
            st.session_state.lstm_history = []
            st.session_state.lstm_message = (
                f"LSTM forecast failed: {e}"
            )

    st.rerun()



if dqn_button:

    try:
        current_state = st.session_state.lstm_current_state
        forecast = st.session_state.lstm_forecast
        risk_result = st.session_state.risk_result

        if current_state is None or forecast is None or risk_result is None:
            raise RuntimeError(
                "Run the LSTM Forecast first so the DQN receives "
                "the current state, 5-step forecast, and risk score."
            )

        encoder = get_state_encoder()

        rl_state = encoder.encode(
            current_state,
            forecast,
            float(risk_result["risk_score"]),
        )

        rl_state = np.asarray(rl_state, dtype=np.float32)

        if rl_state.shape != (20,):
            raise RuntimeError(
                f"Unexpected DQN state shape: {rl_state.shape}; expected (20,)."
            )

        q_values = _dqn_q_values(
            dqn_agent,
            rl_state,
        )

        action = choose_dqn_action(
            dqn_agent,
            rl_state,
        )

        action_name = ACTION_NAMES.get(
            action,
            f"ACTION_{action}",
        )

        with st.spinner(
            f"Applying DQN action: {action_name}..."
        ):
            action_result = apply_dqn_action(action)
            time.sleep(3)

        adapter = CarlaStateAdapter(
            world,
            scenario_tag=(
                f"carla_{st.session_state.active_scenario.lower()}"
            ),
        )

        flags = st.session_state.scenario_flags

        post_state = adapter.get_state(
            timestep=st.session_state.cycle_count,
            accident=flags.get("accident", False),
            road_blockage=flags.get("road_blockage", False),
            emergency_vehicle_count=flags.get("emergency", 0),
        )

        st.session_state.dqn_result = {
            "state_shape": tuple(rl_state.shape),
            "q_values": q_values.tolist(),
            "action": action,
            "action_name": action_name,
            "action_result": action_result,
            "post_state": post_state,
        }

        st.session_state.dqn_message = (
            f"DQN selected {action_name} and applied it to CARLA."
        )

    except Exception as e:

        st.session_state.dqn_result = None
        st.session_state.dqn_message = f"DQN decision failed: {e}"

    st.rerun()


# ============================================================
# STATUS
# ============================================================

if st.session_state.scenario_running:

    st.success(
        f"🟢 Active Scenario: "
        f"{st.session_state.active_scenario}"
    )

else:

    st.info(
        "🔵 CARLA connected — "
        "No active scenario"
    )


# ============================================================
# CURRENT CITY STATE
# ============================================================

state = get_basic_state()

st.subheader(
    "🌆 Current City State"
)

col1, col2, col3, col4, col5, col6 = (
    st.columns(6)
)

with col1:

    st.metric(
        "Vehicles",
        state["vehicles"],
    )

with col2:

    st.metric(
        "Avg Speed",
        f"{state['speed']:.1f} km/h",
    )

with col3:

    st.metric(
        "Traffic Density",
        f"{state['density']:.3f}",
    )

with col4:

    st.metric(
        "Queue",
        f"{state['queue']:.0f} m",
    )

with col5:

    st.metric(
        "Rain",
        f"{state['rain']:.2f}",
    )

with col6:

    st.metric(
        "Fog",
        f"{state['fog']:.2f}",
    )


# ============================================================
# LSTM TRAFFIC FORECAST
# ============================================================

st.divider()

st.subheader("🔮 LSTM Traffic Forecast")

if not lstm_status:

    st.error("Traffic LSTM model failed to load.")
    st.code(lstm_error)

else:

    st.caption(
        f"Input window: {lstm_predictor.seq_len} CARLA observations "
        f"→ Forecast horizon: {lstm_predictor.horizon} steps"
    )

    if st.session_state.lstm_message:

        if st.session_state.lstm_forecast is not None:
            st.success(st.session_state.lstm_message)
        else:
            st.error(st.session_state.lstm_message)

    forecast = st.session_state.lstm_forecast

    if forecast is None:

        st.info(
            "Start a scenario, then click "
            "🔮 Run LSTM Forecast in the sidebar."
        )

    else:

        density_values = forecast["traffic_density"]
        speed_values = forecast["average_speed"]
        queue_values = forecast["queue_length"]

        cols = st.columns(len(density_values))

        for i, col in enumerate(cols):

            with col:

                st.markdown(f"**Step {i + 1}**")

                st.metric(
                    "Density",
                    f"{density_values[i]:.3f}",
                )

                st.metric(
                    "Speed",
                    f"{speed_values[i]:.1f} km/h",
                )

                st.metric(
                    "Queue",
                    f"{queue_values[i]:.0f} m",
                )

        st.markdown("#### Forecast Summary")

        st.dataframe(
            {
                "Forecast Step": list(
                    range(1, len(density_values) + 1)
                ),
                "Traffic Density": [
                    round(value, 4)
                    for value in density_values
                ],
                "Average Speed (km/h)": [
                    round(value, 2)
                    for value in speed_values
                ],
                "Queue Length (m)": [
                    round(value, 2)
                    for value in queue_values
                ],
            },
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "TrafficLSTM forecast from the latest CARLA CityState sequence."
        )


# ============================================================
# URBAN DISRUPTION RISK
# ============================================================

st.divider()

st.subheader("⚠️ Urban Disruption Risk")

if not risk_status:

    st.error("Risk FNN model failed to load.")
    st.code(risk_error)

else:

    risk_result = st.session_state.risk_result

    if risk_result is None:

        st.info(
            "Run the LSTM Forecast to generate the "
            "Urban Disruption Risk assessment."
        )

    else:

        risk_score = float(
            risk_result["risk_score"]
        )

        risk_level = str(
            risk_result["risk_level"]
        )

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f} / 100",
            )

        with risk_col2:

            st.metric(
                "Risk Level",
                risk_level,
            )

        st.progress(
            min(max(risk_score / 100.0, 0.0), 1.0)
        )

        st.caption(
            "Risk FNN uses the latest CARLA CityState "
            "and the 5-step LSTM future trajectory."
        )



# ============================================================
# DQN DECISION / ADAPTIVE RESPONSE
# ============================================================

st.divider()

st.subheader("🤖 DQN Adaptive Response")

if not dqn_status:

    st.error("DQN controller failed to load.")
    st.code(dqn_error)

elif st.session_state.dqn_result is None:

    if st.session_state.lstm_forecast is None:
        st.info(
            "Run the LSTM Forecast first. The DQN uses the current "
            "CARLA state + LSTM forecast + Risk FNN score."
        )
    else:
        st.info(
            "Click 🤖 Run DQN Decision in the sidebar to select and "
            "execute an adaptive traffic-control action in CARLA."
        )

else:

    dqn_result = st.session_state.dqn_result

    if st.session_state.dqn_message:
        st.success(st.session_state.dqn_message)

    dqn_col1, dqn_col2, dqn_col3 = st.columns(3)

    with dqn_col1:
        st.metric(
            "Selected Action",
            dqn_result["action_name"],
        )

    with dqn_col2:
        st.metric(
            "Action ID",
            str(dqn_result["action"]),
        )

    with dqn_col3:
        st.metric(
            "DQN State",
            str(dqn_result["state_shape"]),
        )

    st.markdown("#### Q-Values")

    q_table = {
        "Action": [
            ACTION_NAMES.get(i, f"ACTION_{i}")
            for i in range(len(dqn_result["q_values"]))
        ],
        "Q-Value": [
            round(float(value), 4)
            for value in dqn_result["q_values"]
        ],
    }

    st.dataframe(
        q_table,
        use_container_width=True,
        hide_index=True,
    )

    action_result = dqn_result["action_result"]

    st.markdown("#### CARLA Action Execution")

    if isinstance(action_result, dict):

        result_rows = {
            "Field": list(action_result.keys()),
            "Value": [
                str(value)
                for value in action_result.values()
            ],
        }

        st.dataframe(
            result_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.write(action_result)

    post_state = dqn_result["post_state"]

    st.markdown("#### Post-Action CARLA State")

    post_cols = st.columns(4)

    with post_cols[0]:
        st.metric(
            "Average Speed",
            f"{post_state.average_speed:.2f} km/h",
        )

    with post_cols[1]:
        st.metric(
            "Traffic Density",
            f"{post_state.traffic_density:.3f}",
        )

    with post_cols[2]:
        st.metric(
            "Queue Length",
            f"{post_state.queue_length:.0f} m",
        )

    with post_cols[3]:
        st.metric(
            "Vehicles",
            str(post_state.vehicle_count),
        )

    st.caption(
        "The DQN consumes the 20-dimensional encoded state, selects "
        "one of six learned actions, and the ActionController applies "
        "that action to the CARLA traffic-light control layer."
    )


# ============================================================
# LIVE CAMERA + CNN PERCEPTION
# ============================================================

st.divider()

st.subheader(
    "📷 Live Urban Perception"
)


@st.fragment(run_every="1s")
def render_camera_and_cnn():

    frame, frame_number = (
        camera_buffer.get()
    )

    # --------------------------------------------------------
    # No frame yet
    # --------------------------------------------------------

    if frame is None:

        if st.session_state.scenario_running:

            st.info(
                "Camera is active. "
                "Waiting for first frame..."
            )

        else:

            st.info(
                "Start a scenario to activate "
                "the CARLA camera."
            )

        return

    # --------------------------------------------------------
    # CAMERA + CNN COLUMNS
    # --------------------------------------------------------

    camera_col, cnn_col = st.columns(
        [1.55, 1]
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    with camera_col:

        st.markdown(
            "### 🎥 CARLA Camera"
        )

        st.image(
            frame,
            channels="RGB",
            width=CAMERA_WIDTH,
        )

        st.caption(
            f"CARLA frame: {frame_number}"
        )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    with cnn_col:

        st.markdown(
            "### 🧠 CNN Perception"
        )

        if not cnn_status:

            st.error(
                "CNN model failed to load."
            )

            st.code(
                cnn_error
            )

            return

        try:

            prediction = (
                cnn_predictor.predict(frame)
            )

            vehicle_prob = prediction.get(
                "vehicle_present",
                0.0,
            )

            pedestrian_prob = prediction.get(
                "pedestrian_present",
                0.0,
            )

            obstruction_prob = prediction.get(
                "road_obstruction",
                0.0,
            )

            visibility_prob = prediction.get(
                "low_visibility",
                0.0,
            )

            # ------------------------------------------------
            # VEHICLE
            # ------------------------------------------------

            st.write(
                f"🚗 **Vehicle Present** — "
                f"{vehicle_prob * 100:.1f}%"
            )

            st.progress(
                float(vehicle_prob)
            )

            # ------------------------------------------------
            # PEDESTRIAN
            # ------------------------------------------------

            st.write(
                f"🚶 **Pedestrian Present** — "
                f"{pedestrian_prob * 100:.1f}%"
            )

            st.progress(
                float(pedestrian_prob)
            )

            # ------------------------------------------------
            # OBSTRUCTION
            # ------------------------------------------------

            st.write(
                f"🚧 **Road Obstruction** — "
                f"{obstruction_prob * 100:.1f}%"
            )

            st.progress(
                float(obstruction_prob)
            )

            # ------------------------------------------------
            # LOW VISIBILITY
            # ------------------------------------------------

            st.write(
                f"🌫️ **Low Visibility** — "
                f"{visibility_prob * 100:.1f}%"
            )

            st.progress(
                float(visibility_prob)
            )

            st.divider()

            st.caption(
                "UrbanPerceptionCNN • "
                "4-label multi-label inference"
            )

            st.caption(
                "Input: 128 × 128 RGB • CPU inference"
            )

        except Exception as e:

            st.error(
                "CNN inference failed."
            )

            st.exception(e)


render_camera_and_cnn()


# ============================================================
# SCENARIO CONDITIONS
# ============================================================

st.divider()

st.subheader(
    "⚠️ Scenario Conditions"
)

c1, c2, c3 = st.columns(3)

flags = st.session_state.scenario_flags

with c1:

    st.metric(
        "Accident",
        "YES"
        if flags.get("accident")
        else "NO",
    )

with c2:

    st.metric(
        "Road Blockage",
        "YES"
        if flags.get("road_blockage")
        else "NO",
    )

with c3:

    st.metric(
        "Emergency Vehicles",
        flags.get("emergency", 0),
    )


# ============================================================
# CNN OBSERVABLES
# ============================================================

st.divider()

st.subheader(
    "🔎 Visual Urban Observables"
)


@st.fragment(run_every="1s")
def render_observables():

    frame, _ = camera_buffer.get()

    if frame is None:

        st.info(
            "Visual observables will appear "
            "when the camera is active."
        )

        return

    if not cnn_status:

        return

    try:

        observables = (
            cnn_predictor.predict_observables(
                frame
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Visual Vehicle",
                f"{observables['visual_vehicle_indicator']:.3f}",
            )

        with c2:

            st.metric(
                "Visual Pedestrian",
                f"{observables['visual_pedestrian_indicator']:.3f}",
            )

        with c3:

            st.metric(
                "Visual Obstruction",
                f"{observables['visual_obstruction_indicator']:.3f}",
            )

        with c4:

            st.metric(
                "Visual Visibility",
                f"{observables['visual_visibility_indicator']:.3f}",
            )

    except Exception as e:

        st.error(
            "Observable extraction failed."
        )

        st.exception(e)


render_observables()


# ============================================================
# SIMULATION STATUS
# ============================================================

st.divider()

st.subheader(
    "⚙️ Simulation Status"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Scenario",
        st.session_state.active_scenario
        or "None",
    )

with c2:

    st.metric(
        "AI Cycles",
        st.session_state.cycle_count,
    )

with c3:

    st.metric(
        "CARLA Vehicles",
        len(
            world.get_actors().filter(
                "vehicle.*"
            )
        ),
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CityTwin AI • CARLA-based Urban Digital Twin"
)