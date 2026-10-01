"""
rl/replay_buffer.py — Experience Replay Buffer for DQN
======================================================
Stores transitions (s, a, r, s', done) and samples minibatches
for off-policy Q-learning.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (default 10,000).
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition."""
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random minibatch of transitions.

        Returns
        -------
        states : Tensor of shape (batch_size, state_dim)
        actions : Tensor of shape (batch_size, 1)
        rewards : Tensor of shape (batch_size, 1)
        next_states : Tensor of shape (batch_size, state_dim)
        dones : Tensor of shape (batch_size, 1)
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(np.array([t.reward for t in batch]), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
