"""
rl/agent.py — Deep Q-Network (DQN) Agent & Baseline Policies
============================================================
Implements:
  1. QNetwork (PyTorch MLP: state_dim -> 64 -> 64 -> action_dim)
  2. DQNAgent (Target Network, Epsilon-Greedy, Experience Replay, Adam)
  3. NoActionPolicy (Baseline 1: Always 0)
  4. HeuristicPolicy (Baseline 2: Domain-expert rule system)
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.state_encoder import STATE_DIM
from rl.environment import TrafficAction, ACTION_DIM
from rl.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Q-Network Architecture
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Q-value estimator mapping state_dim (20) to Q-values for all 6 discrete actions.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, state_dim)

        Returns
        -------
        Tensor of shape (batch_size, action_dim)
        """
        return self.net(x)

    def _init_weights(self) -> None:
        """Kaiming normal initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network Agent with Target Network and Epsilon-Greedy Exploration.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action via epsilon-greedy strategy.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        self.policy_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s_t)
            action = int(q_values.argmax(dim=1).item())

        return action

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 64,
    ) -> Optional[float]:
        """
        Perform one gradient descent step on a minibatch from replay buffer.
        """
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device=self.device)

        # Q(s, a)
        self.policy_net.train()
        q_current = self.policy_net(states).gather(1, actions)

        # Target Q = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + (1.0 - dones) * self.gamma * q_next_max

        loss = self.criterion(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
            },
            str(p),
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        p = Path(path)
        ckpt = torch.load(str(p), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net_state_dict"])
        self.update_target_network()
        self.policy_net.eval()


# ---------------------------------------------------------------------------
# Baseline Policies
# ---------------------------------------------------------------------------

class NoActionPolicy:
    """Baseline 1: Always select NO_ACTION (0)."""

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        return TrafficAction.NO_ACTION


class HeuristicPolicy:
    """
    Baseline 2: Domain-expert rule-based traffic controller.

    Rules:
      1. If active emergency vehicles and high congestion/blockage -> COMBINED_RESPONSE
      2. If active emergency vehicles -> EMERGENCY_PRIORITY
      3. If high disruption risk (>= 70) or road blockage -> TRAFFIC_DIVERSION
      4. If queue is growing or density is high (> 0.65) -> EXTEND_GREEN
      5. If traffic is very light (< 0.20) and speed is high -> REDUCE_GREEN
      6. Otherwise -> NO_ACTION
    """

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        # State vector indices from StateEncoder:
        # 0: density, 1: speed, 2: queue, 7: blockage, 8: accident, 9: ev_count, 16: risk, 19: queue_growth
        density = state[0]
        speed = state[1]
        blockage = state[7]
        ev_count = state[9]
        risk = state[16] * 100.0
        queue_growth = state[19]

        if ev_count > 0.0 and (blockage > 0.5 or density > 0.60):
            return TrafficAction.COMBINED_RESPONSE
        elif ev_count > 0.0:
            return TrafficAction.EMERGENCY_PRIORITY
        elif risk >= 70.0 or blockage > 0.5:
            return TrafficAction.TRAFFIC_DIVERSION
        elif queue_growth > 0.05 or density > 0.65:
            return TrafficAction.EXTEND_GREEN
        elif density < 0.20 and speed > 0.50:
            return TrafficAction.REDUCE_GREEN
        else:
            return TrafficAction.NO_ACTION
