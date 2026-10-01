"""
perception/predictor.py — High-Level Urban Perception Predictor
================================================================
Provides a clean standalone inference API for camera-based visual scene
perception using the trained UrbanPerceptionCNN.

Usage
-----
>>> from perception.predictor import UrbanPerceptionPredictor
>>> predictor = UrbanPerceptionPredictor.from_checkpoint("models/urban_cnn.pt")
>>> result = predictor.predict(camera_image)
>>> print(result)
{
    "vehicle_present": 0.9421,
    "pedestrian_present": 0.0815,
    "road_obstruction": 0.0342,
    "low_visibility": 0.0120
}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from perception.synthetic_data import LABEL_NAMES, NUM_LABELS
from perception.cnn_model import UrbanPerceptionCNN
from perception.feature_extractor import (
    preprocess_image,
    perception_to_observables,
    UrbanPerceptionFeatureExtractor,
)


class UrbanPerceptionPredictor:
    """
    Inference interface wrapping the trained UrbanPerceptionCNN model.
    """

    def __init__(
        self,
        model: UrbanPerceptionCNN,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._extractor = UrbanPerceptionFeatureExtractor(self._model, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = _ROOT / "models" / "urban_cnn.pt",
        device: str = "cpu",
    ) -> "UrbanPerceptionPredictor":
        """
        Instantiate UrbanPerceptionPredictor from a saved checkpoint file.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Urban CNN checkpoint not found: {path}")

        dev = torch.device(device)
        ckpt = torch.load(str(path), map_location=dev, weights_only=False)

        config = ckpt.get("model_config", {})
        model = UrbanPerceptionCNN(
            in_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", NUM_LABELS),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev)
        model.eval()

        return cls(model=model, device=dev)

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Predict detection probabilities for each of the 4 urban scene labels.

        Parameters
        ----------
        image : np.ndarray (128x128x3) or PIL.Image or Tensor (3x128x128)

        Returns
        -------
        dict mapping label names to probabilities in [0, 1]
        """
        return self._extractor.predict_probabilities(image)

    def extract_features(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
    ) -> np.ndarray:
        """
        Extract the intermediate 32-dimensional feature embedding before classification.
        """
        return self._extractor.extract_features(image)

    def predict_observables(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Predict and map directly to normalized city state perception observables.
        """
        probs = self.predict(image)
        return perception_to_observables(probs)

    @property
    def model(self) -> UrbanPerceptionCNN:
        """Access underlying PyTorch CNN model."""
        return self._model

    def __repr__(self) -> str:
        return f"UrbanPerceptionPredictor(device={self._device}, params={self._model.count_parameters():,})"
