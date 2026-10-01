"""
lstm_model.py — TrafficLSTM
============================
Lightweight PyTorch LSTM that sits at the heart of CityTwin AI's
temporal prediction unit (Unit III).

Architecture
------------

  Input: (batch, seq_len, input_size)        ← normalised feature vectors
         ↓
  LSTM (hidden=64, layers=2, dropout=0.2)
         ↓
  Take final hidden state: (batch, hidden_size)
         ↓
  LayerNorm
         ↓
  Dropout(0.2)
         ↓
  Fully-connected: hidden_size → hidden_size // 2
         ↓
  ReLU
         ↓
  Fully-connected: hidden_size // 2 → horizon * target_dim
         ↓
  Reshape: (batch, horizon, target_dim)
         ↓
  Sigmoid (keep outputs in [0, 1] — targets are normalised)

CityTwin pipeline position
--------------------------
  CityState sequence  →  FeatureExtractor  →  TrafficLSTM  →  predictions
                                                     ↓
                                          {traffic_density,
                                           average_speed,
                                           queue_length}  (normalised)
                                                     ↓
                                          FeatureExtractor.inverse_targets()
                                                     ↓
                                          Real-world units (km/h, metres)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """
    Sequence-to-sequence LSTM for urban traffic prediction.

    Parameters
    ----------
    input_size : int
        Number of features per timestep (= FEATURE_DIM, default 14).
    hidden_size : int
        LSTM hidden dimension (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    horizon : int
        Number of future timesteps to predict (default 5).
    target_dim : int
        Number of output variables per future timestep (default 3:
        traffic_density, average_speed, queue_length).
    dropout : float
        Dropout probability applied between LSTM layers and before the
        final FC layer (default 0.2).

    Forward
    -------
    Input : (batch, seq_len, input_size)
    Output: (batch, horizon, target_dim)  — values in [0, 1]
    """

    def __init__(
        self,
        input_size:  int = 14,
        hidden_size: int = 64,
        num_layers:  int = 2,
        horizon:     int = 5,
        target_dim:  int = 3,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.horizon     = horizon
        self.target_dim  = target_dim

        # LSTM — inter-layer dropout only applies when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Post-LSTM head
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        half = hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, horizon * target_dim),
        )

        self.output_act = nn.Sigmoid()

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: "torch.Tensor",
        hidden: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, input_size)
        hidden : optional initial (h_0, c_0) — if None, zeros are used.

        Returns
        -------
        Tensor, shape (batch, horizon, target_dim)
            All values in [0, 1] (sigmoid-activated).
        """
        out, _ = self.lstm(x, hidden)          # (batch, seq_len, hidden)
        last   = out[:, -1, :]                 # (batch, hidden) — last step
        last   = self.norm(last)
        last   = self.dropout(last)
        pred   = self.head(last)               # (batch, horizon * target_dim)
        pred   = pred.view(-1, self.horizon, self.target_dim)
        return self.output_act(pred)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for LSTM weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"TrafficLSTM("
            f"input={self.input_size}, "
            f"hidden={self.hidden_size}, "
            f"layers={self.num_layers}, "
            f"horizon={self.horizon}, "
            f"targets={self.target_dim}, "
            f"params={self.count_parameters():,})"
        )
