"""
perception/cnn_model.py — Lightweight Urban CNN Architecture
=============================================================
Convolutional Neural Network for multi-label urban scene perception.

Architecture
------------
  Input: (batch, 3, 128, 128)
    │
  Conv2D(3 → 16, k=3, p=1) ──► ReLU ──► MaxPool2D(2)    (16 × 64 × 64)
    │
  Conv2D(16 → 32, k=3, p=1) ──► ReLU ──► MaxPool2D(2)   (32 × 32 × 32)
    │
  Conv2D(32 → 64, k=3, p=1) ──► ReLU ──► MaxPool2D(2)   (64 × 16 × 16)
    │
  AdaptiveAvgPool2D((1, 1))                             (64 × 1 × 1)
    │
  Flatten                                               (64)
    │
  Linear(64 → 32) ──► ReLU ──► Dropout(0.2)             (32)  <-- Feature Embedding
    │
  Linear(32 → 4)                                        (4)   <-- Logits Output
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from perception.synthetic_data import NUM_LABELS


class UrbanPerceptionCNN(nn.Module):
    """
    Lightweight CNN for Multi-Label Urban Scene Perception.

    Parameters
    ----------
    in_channels : int (default 3)
    num_classes : int (default 4)
    dropout : float (default 0.2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_LABELS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.AdaptiveAvgPool2d((1, 1)),  # 64x1x1
        )

        # Embedding & classifier head
        self.fc_embed = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(32, num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.

        Parameters
        ----------
        x : Tensor of shape (batch, 3, 128, 128)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        feats = self.features(x)
        flat = torch.flatten(feats, 1)
        embed = self.fc_embed(flat)
        logits = self.classifier(embed)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 32-dimensional intermediate feature representation.

        Parameters
        ----------
        x : Tensor of shape (batch, 3, 128, 128)

        Returns
        -------
        embedding : Tensor of shape (batch, 32)
        """
        feats = self.features(x)
        flat = torch.flatten(feats, 1)
        return self.fc_embed(flat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid probabilities for each label in [0, 1].

        Returns
        -------
        probs : Tensor of shape (batch, num_classes)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UrbanPerceptionCNN("
            f"in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
