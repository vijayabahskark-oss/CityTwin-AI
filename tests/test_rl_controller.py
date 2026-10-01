"""
tests/test_rl_controller.py
============================
Comprehensive test suite for Stage 4: Reinforcement Learning Adaptive Traffic Controller.

Coverage
--------
  1. test_env_initialization
  2. test_env_reset
  3. test_state_encoder_shape_and_bounds
  4. test_all_actions_execution
  5. test_step_returns_and_types
  6. test_reward_finiteness
  7. test_deterministic_reset
  8. test_dqn_qnetwork_forward
  9. test_replay_buffer_operations
  10. test_epsilon_greedy_behavior
  11. test_target_network_sync
  12. test_dqn_training_step
  13. test_dqn_save_load_roundtrip
  14. test_baseline_policies
  15. test_short_rl_episode
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pytest

from simulation.scenario_generator import ScenarioType
from simulation.mock_city import CityState
from rl.state_encoder import StateEncoder, STATE_DIM, STATE_FEATURE_NAMES
from rl.environment import CityTrafficEnv, TrafficAction, ACTION_DIM, ACTION_DESCRIPTIONS
from rl.replay_buffer import ReplayBuffer
from rl.agent import QNetwork, DQNAgent, NoActionPolicy, HeuristicPolicy


# ---------------------------------------------------------------------------
# Test 1: Environment Initialization
# ---------------------------------------------------------------------------

def test_env_initialization():
    env = CityTrafficEnv()
    assert env.action_space.n == ACTION_DIM
    assert env.observation_space.shape == (STATE_DIM,)
    assert len(ACTION_DESCRIPTIONS) == ACTION_DIM


# ---------------------------------------------------------------------------
# Test 2: Environment Reset
# ---------------------------------------------------------------------------

def test_env_reset():
    env = CityTrafficEnv(scenario=ScenarioType.NORMAL, seed=42)
    state, info = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    assert isinstance(info, dict)
    assert "scenario" in info
    assert "vehicle_count" in info


# ---------------------------------------------------------------------------
# Test 3: State Encoder Shape & Bounds
# ---------------------------------------------------------------------------

def test_state_encoder_shape_and_bounds():
    encoder = StateEncoder()
    sim_state = CityState(
        vehicle_count=150,
        average_speed=45.0,
        traffic_density=0.35,
        queue_length=60.0,
        pedestrian_count=40,
        rain_intensity=0.1,
        fog_intensity=0.05,
        temperature=22.0,
        road_blockage=False,
        accident=False,
        emergency_vehicle_count=1,
        traffic_signal_state="green",
        timestamp="2026-09-29T12:00:00Z",
        timestep=0,
    )
    forecast = {
        "traffic_density": [0.35, 0.38, 0.40, 0.42, 0.45],
        "average_speed": [45.0, 42.0, 40.0, 38.0, 35.0],
        "queue_length": [60.0, 75.0, 90.0, 110.0, 130.0],
    }
    risk_score = 35.5

    vec = encoder.encode(sim_state, forecast, risk_score)
    assert vec.shape == (STATE_DIM,)
    assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# Test 4: All 6 Actions Execution
# ---------------------------------------------------------------------------

def test_all_actions_execution():
    env = CityTrafficEnv(seed=12)
    env.reset()

    for action in TrafficAction:
        next_state, reward, terminated, truncated, info = env.step(action)
        assert next_state.shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert not terminated
        assert info["action"] == action


# ---------------------------------------------------------------------------
# Test 5: Step Returns and Types
# ---------------------------------------------------------------------------

def test_step_returns_and_types():
    env = CityTrafficEnv(max_steps=5, seed=10)
    env.reset()

    for step in range(5):
        next_state, reward, terminated, truncated, info = env.step(TrafficAction.EXTEND_GREEN)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    assert truncated  # Hit max_steps


# ---------------------------------------------------------------------------
# Test 6: Reward Finiteness
# ---------------------------------------------------------------------------

def test_reward_finiteness():
    env = CityTrafficEnv(scenario=ScenarioType.ACCIDENT_RAIN, seed=99)
    env.reset()

    for _ in range(10):
        action = np.random.randint(0, ACTION_DIM)
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# Test 7: Deterministic Reset with Seed
# ---------------------------------------------------------------------------

def test_deterministic_reset():
    env1 = CityTrafficEnv(seed=77)
    env2 = CityTrafficEnv(seed=77)

    s1, _ = env1.reset(seed=77)
    s2, _ = env2.reset(seed=77)

    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Test 8: DQN QNetwork Forward Pass
# ---------------------------------------------------------------------------

def test_dqn_qnetwork_forward():
    net = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
    net.eval()

    batch_size = 4
    x = torch.rand(batch_size, STATE_DIM)
    with torch.no_grad():
        q_vals = net(x)

    assert q_vals.shape == (batch_size, ACTION_DIM)
    assert torch.all(torch.isfinite(q_vals))


# ---------------------------------------------------------------------------
# Test 9: Replay Buffer Operations
# ---------------------------------------------------------------------------

def test_replay_buffer_operations():
    buffer = ReplayBuffer(capacity=100)
    assert len(buffer) == 0

    s = np.zeros(STATE_DIM, dtype=np.float32)
    s_next = np.ones(STATE_DIM, dtype=np.float32)

    for i in range(10):
        buffer.push(s, i % 6, 1.0, s_next, False)

    assert len(buffer) == 10
    states, actions, rewards, next_states, dones = buffer.sample(4)

    assert states.shape == (4, STATE_DIM)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, STATE_DIM)
    assert dones.shape == (4, 1)


# ---------------------------------------------------------------------------
# Test 10: Epsilon Greedy Action Selection
# ---------------------------------------------------------------------------

def test_epsilon_greedy_behavior():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # Greedy (epsilon = 0.0) -> Deterministic
    a1 = agent.select_action(state, epsilon=0.0)
    a2 = agent.select_action(state, epsilon=0.0)
    assert a1 == a2
    assert 0 <= a1 < ACTION_DIM


# ---------------------------------------------------------------------------
# Test 11: Target Network Sync
# ---------------------------------------------------------------------------

def test_target_network_sync():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # Modify policy net weights
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.add_(1.0)

    agent.update_target_network()

    for p_pol, p_tgt in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_pol, p_tgt)


# ---------------------------------------------------------------------------
# Test 12: DQN Training Iteration
# ---------------------------------------------------------------------------

def test_dqn_training_step():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(capacity=100)

    s = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(20):
        buffer.push(s, 0, 1.0, s, False)

    loss = agent.update(buffer, batch_size=8)
    assert loss is not None
    assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 13: Model Save & Load Roundtrip
# ---------------------------------------------------------------------------

def test_dqn_save_load_roundtrip():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
    state = np.random.rand(STATE_DIM).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_rl.pt"
        agent.save(path)

        loaded_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=32)
        loaded_agent.load(path)

    # Compare Q-values
    with torch.no_grad():
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q1 = agent.policy_net(s_t).numpy()
        q2 = loaded_agent.policy_net(s_t).numpy()

    np.testing.assert_allclose(q1, q2, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 14: Baseline Policies
# ---------------------------------------------------------------------------

def test_baseline_policies():
    no_act = NoActionPolicy()
    heur = HeuristicPolicy()

    normal_state = np.zeros(STATE_DIM, dtype=np.float32)
    assert no_act.select_action(normal_state) == TrafficAction.NO_ACTION

    # Emergency state (index 9 = ev_count)
    emg_state = np.zeros(STATE_DIM, dtype=np.float32)
    emg_state[9] = 0.3  # EV present
    assert heur.select_action(emg_state) == TrafficAction.EMERGENCY_PRIORITY

    # Critical risk state (index 16 = risk/100)
    risk_state = np.zeros(STATE_DIM, dtype=np.float32)
    risk_state[16] = 0.85
    assert heur.select_action(risk_state) == TrafficAction.TRAFFIC_DIVERSION


# ---------------------------------------------------------------------------
# Test 15: Short RL Episode Loop
# ---------------------------------------------------------------------------

def test_short_rl_episode():
    env = CityTrafficEnv(scenario=ScenarioType.NORMAL, max_steps=10, seed=42)
    agent = DQNAgent()
    state, _ = env.reset()

    total_reward = 0.0
    for _ in range(10):
        action = agent.select_action(state, epsilon=0.1)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    assert np.isfinite(total_reward)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
