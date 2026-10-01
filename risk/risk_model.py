"""
risk/risk_model.py — Feedforward Neural Network for Risk Assessment
====================================================================
Implements a lightweight Multi-Layer Perceptron (MLP) mapping the
23-dimensional fused feature representation to a single continuous
urban disruption risk score.

Architecture
------------
  Input (batch, 23)
    │
  Linear(23 → 64)
    │
  ReLU
    │
  Dropout(0.15)
    │
  Linear(64 → 32)
    │
  ReLU
    │
  Dropout(0.10)
    │
  Linear(32 → 1)
    │
  Sigmoid()
    │
  Output (batch, 1) ∈ [0, 1]  ──(× 100)──► Risk Score ∈ [0, 100]
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn

from risk.feature_builder import RISK_FEATURE_DIM


class RiskMLP(nn.Module):
    """
    Feedforward Neural Network for Urban Disruption Severity / Risk Assessment.

    Parameters
    ----------
    input_dim : int
        Number of fused input features (default 23).
    hidden_dims : list of int
        Sizes of hidden layers (default [64, 32]).
    dropout : float
        Dropout probability (default 0.15).
    """

    def __init__(
        self,
        input_dim: int = RISK_FEATURE_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Sequential MLP layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7 if dropout > 0 else 0.0),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch_size, 1), normalized values in [0, 1]
        """
        return self.net(x)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute risk score scaled to [0, 100].

        Returns
        -------
        torch.Tensor of shape (batch_size, 1), values in [0, 100]
        """
        return self.forward(x) * 100.0

    def _init_weights(self) -> None:
        """Kaiming uniform for ReLU linear layers, Xavier for output."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"RiskMLP(input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim=1, "
            f"params={self.count_parameters():,})"
        )
