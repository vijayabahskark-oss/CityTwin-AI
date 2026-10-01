# CityTwin AI — Predictive Urban Monitoring and Adaptive Response

CityTwin AI is a prototype urban digital-twin system that combines simulated
city observations, computer vision, temporal traffic forecasting, disruption
risk assessment, and reinforcement learning.

It uses CARLA 0.9.16 for simulated urban scenes and traffic, PyTorch for
machine-learning models, and Streamlit for interactive visualization.

The project explores a closed-loop workflow:

**OBSERVE → PREDICT → ASSESS → DECIDE → ACT → OBSERVE AGAIN**

This is a city-scale monitoring and traffic-response prototype, not an
autonomous-driving system.

## Architecture

1. **Simulation and observation:** Obtain urban state observations from the
   mock simulator or CARLA integration.
2. **Visual perception (CNN):** Estimate the presence of vehicles,
   pedestrians, road obstructions, and low visibility from an image.
3. **Temporal prediction (LSTM):** Process a sequence of 10 observations to
   forecast traffic density, average speed, and queue length over a five-step
   horizon.
4. **Risk assessment (FNN):** Combine current-state features, forecast
   summaries, and trends to produce an Urban Disruption Risk Score from 0 to
   100.
5. **Adaptive response (DQN):** Evaluate six candidate traffic-control
   actions using a 20-dimensional state representation.
6. **Action execution:** Apply the selected action through the current CARLA
   traffic-signal controller implementation.
7. **Dashboard:** Visualize observations, predictions, risk, and controller
   activity.

## Implementation status

| Component | Status |
|---|---|
| Mock city simulation and scenario generation | Implemented |
| LSTM temporal traffic predictor | Implemented |
| Feedforward disruption-risk assessor | Implemented |
| DQN traffic controller | Implemented |
| CNN visual perception model | Implemented |
| CARLA 0.9.16 state integration | Implemented |
| Streamlit dashboard | Implemented |
| Final end-to-end validation and broader benchmarking | Ongoing |

## Models

### LSTM traffic predictor

- Input: 14 features per observation.
- Historical sequence length: 10 observations.
- Forecast horizon: 5 steps.
- Architecture: two LSTM layers with hidden size 64.
- Predicted quantities: traffic density, average speed, and queue length.

### Feedforward risk assessor

- Input: 23 features comprising current-state features, forecast summaries,
  and trends.
- Architecture: 23 → 64 → 32 → 1, with a sigmoid output.
- Output: a prototype Urban Disruption Risk Score from 0 to 100, categorized
  into LOW, MEDIUM, HIGH, or CRITICAL risk levels.

The score is a model output, not a calibrated real-world accident probability.

### CNN visual perception

- Input: RGB image resized to 128 × 128.
- Multi-label outputs: vehicle presence, pedestrian presence, road obstruction,
  and low visibility.
- The current model was trained using synthetic scenes. Its predictions should
  be treated as prototype estimates, not validated real-world detections.

### DQN adaptive controller

The controller evaluates six actions:

| Action ID | Action |
|---:|---|
| 0 | NO_ACTION |
| 1 | EXTEND_GREEN |
| 2 | REDUCE_GREEN |
| 3 | TRAFFIC_DIVERSION |
| 4 | EMERGENCY_PRIORITY |
| 5 | COMBINED_RESPONSE |

The current CARLA action-controller implementation is an MVP: it changes a
traffic light's phase for selected actions. Traffic diversion is not full
route planning, and the signal mapping is not a complete traffic-management
system.

## CARLA scenarios

The integration supports prototype scenario configurations covering:

- Normal traffic
- Rush hour
- Heavy rain
- Fog
- Accident
- Road blockage
- Emergency response
- Heavy rain combined with an accident

Some accident and blockage conditions are represented as scenario/state flags;
they do not necessarily create a physically simulated collision or road barrier.

## Repository structure

```text
CityTwin-AI/
├── carla_integration/   # CARLA state adapter and action integration
├── models/              # Model checkpoints and LSTM scaler parameters
├── perception/          # CNN visual perception
├── prediction/          # LSTM traffic forecasting
├── risk/                # Disruption-risk assessment
├── rl/                  # DQN controller and environment
├── simulation/          # Mock city simulator and scenarios
├── tests/               # Automated tests
├── dashboard_complete.py
├── requirements.txt
└── README.md
```

## Requirements

- Windows 64-bit is the tested CARLA setup.
- Python 3.12 was used for the CARLA 0.9.16 integration.
- PyTorch CPU is sufficient for the current prototype.
- CARLA 0.9.16 must be installed separately.
- The Streamlit dashboard uses the project's Python dependencies and requires
  a running CARLA server for live CARLA functionality.

## Setup

### 1. Create a virtual environment

From the project directory:

```powershell
py -3.12 -m venv .venv_dashboard
.\.venv_dashboard\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Install the CARLA Python API

Install and extract CARLA 0.9.16 separately. If using the Windows wheel
distributed with the CARLA package, install the wheel matching your Python
version and system architecture. For example, with the CARLA installation
located at `D:\CARLA_0.9.16`:

```powershell
python -m pip install "D:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
```

Adjust the path and wheel filename if your installation differs.

### 4. Start the CARLA server

For the example installation path:

```powershell
cd D:\CARLA_0.9.16
.\CarlaUE4.exe
```

Wait for the simulator to finish loading before starting live dashboard
operations.

### 5. Launch the dashboard

In a second PowerShell window:

```powershell
cd "D:\CityTwin AI"
.\.venv_dashboard\Scripts\Activate.ps1
streamlit run dashboard_complete.py
```

Use the local URL printed by Streamlit to open the dashboard.

## Testing

Run the project's available tests from the project root with the project
environment activated:

```powershell
python -m pytest -q
```

Individual test scripts are also available in the repository. Some CARLA
integration tests require a running CARLA server; tests that require external
simulation services may not run in a clean environment without additional setup.

## Model checkpoints

The `models/` directory contains the trained prototype checkpoints and scaler
parameters required by model inference. These files are small enough to keep
with the source repository.

Keep model files and source code versioned together so that the model-loading
paths remain consistent.

## Limitations

- The CNN uses synthetic training images, creating a domain gap from CARLA
  imagery and real-world camera feeds.
- The risk score is not calibrated against real-world incident probabilities.
- The DQN was trained in a mock environment; its behavior in CARLA does not
  establish real-world effectiveness or safety.
- The CARLA traffic-state adapter uses simplified measurements and assumptions.
- Current action execution is a signal-control MVP, not full lane-level
  rerouting or city-wide signal coordination.
- Accident and road-blockage scenarios may be represented by flags rather than
  physical collision and obstruction simulation.
- Performance and behavior require broader validation across scenarios and
  random seeds before drawing general conclusions.

## Project goal

CityTwin AI is a research and educational prototype for exploring how
perception, temporal forecasting, risk assessment, and reinforcement learning
can be connected in a simulated urban digital twin.
