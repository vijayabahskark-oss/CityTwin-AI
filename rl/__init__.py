"""
CityTwin AI — Reinforcement Learning Package (Stage 4: Adaptive Traffic Controller)
====================================================================================
Implements a Gymnasium-compatible urban traffic management environment and
a Deep Q-Network (DQN) agent that selects closed-loop city-level interventions.

Public API
----------
  CityTrafficEnv       — Gymnasium-compatible environment
  TrafficAction        — Enum of 6 discrete traffic control actions
  ACTION_DESCRIPTIONS  — Human-readable action documentation
  ACTION_DIM           — Number of discrete actions (6)
  StateEncoder         — 20-dim state encoder (Observables + Forecast + Risk + Trends)
  STATE_DIM            — State vector dimension (20)
  STATE_FEATURE_NAMES  — Names of state features
  QNetwork             — PyTorch Q-value neural network
  DQNAgent             — Deep Q-Network agent with experience replay and target network
  NoActionPolicy       — Baseline 1 (Always action 0)
  HeuristicPolicy      — Baseline 2 (Expert rule-based policy)
  ReplayBuffer         — Fixed-capacity experience replay memory
"""

from .state_encoder import StateEncoder, STATE_DIM, STATE_FEATURE_NAMES
from .environment import (
    CityTrafficEnv,
    TrafficAction,
    ACTION_DESCRIPTIONS,
    ACTION_DIM,
    ACTION_COSTS,
)
from .replay_buffer import ReplayBuffer
from .agent import (
    QNetwork,
    DQNAgent,
    NoActionPolicy,
    HeuristicPolicy,
)

__all__ = [
    "CityTrafficEnv",
    "TrafficAction",
    "ACTION_DESCRIPTIONS",
    "ACTION_DIM",
    "ACTION_COSTS",
    "StateEncoder",
    "STATE_DIM",
    "STATE_FEATURE_NAMES",
    "QNetwork",
    "DQNAgent",
    "NoActionPolicy",
    "HeuristicPolicy",
    "ReplayBuffer",
]
