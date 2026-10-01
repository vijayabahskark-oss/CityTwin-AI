"""
perception/feature_extractor.py — CNN Feature Extraction & Observable Mapping
============================================================================
Exposes helper methods to extract intermediate 32-dim feature embeddings
from camera images and map multi-label probabilities into normalized
urban perception observables for downstream sensor fusion.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.synthetic_data import LABEL_NAMES, NUM_LABELS, IMAGE_SIZE
from perception.cnn_model import UrbanPerceptionCNN


def preprocess_image(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert image into a normalized (1, 3, 128, 128) float32 PyTorch tensor.

    Parameters
    ----------
    image : np.ndarray (H,W,3) or PIL.Image or Tensor (3,H,W) / (1,3,H,W)

    Returns
    -------
    torch.Tensor of shape (1, 3, 128, 128) on specified device.
    """
    if isinstance(image, Image.Image):
        img_resized = image.resize(IMAGE_SIZE)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        # HWC -> CHW -> 1CHW
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
    elif isinstance(image, np.ndarray):
        if image.shape[:2] != IMAGE_SIZE:
            pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
            pil_img = pil_img.resize(IMAGE_SIZE)
            arr = np.array(pil_img, dtype=np.float32)
        else:
            arr = image.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 3 and arr.shape[-1] == 3:
            # (H, W, 3) -> (3, H, W)
            arr = np.transpose(arr, (2, 0, 1))

        if arr.ndim == 3:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            tensor = torch.from_numpy(arr)
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return tensor.to(device)


def perception_to_observables(
    prediction: Dict[str, float],
) -> Dict[str, float]:
    """
    Map raw multi-label detection probabilities into normalized urban observables.

    Parameters
    ----------
    prediction : dict with keys 'vehicle_present', 'pedestrian_present', etc.

    Returns
    -------
    dict with normalized indicators:
      - 'visual_vehicle_indicator'
      - 'visual_pedestrian_indicator'
      - 'visual_obstruction_indicator'
      - 'visual_visibility_indicator'
    """
    v_prob = float(np.clip(prediction.get("vehicle_present", 0.0), 0.0, 1.0))
    p_prob = float(np.clip(prediction.get("pedestrian_present", 0.0), 0.0, 1.0))
    o_prob = float(np.clip(prediction.get("road_obstruction", 0.0), 0.0, 1.0))
    l_prob = float(np.clip(prediction.get("low_visibility", 0.0), 0.0, 1.0))

    return {
        "visual_vehicle_indicator": round(v_prob, 4),
        "visual_pedestrian_indicator": round(p_prob, 4),
        "visual_obstruction_indicator": round(o_prob, 4),
        "visual_visibility_indicator": round(1.0 - l_prob, 4),  # 1.0 = clear, 0.0 = zero visibility
    }


class UrbanPerceptionFeatureExtractor:
    """
    Extracts intermediate embeddings and predictions from camera images using UrbanPerceptionCNN.
    """

    def __init__(
        self,
        model: UrbanPerceptionCNN,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def extract_features(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
    ) -> np.ndarray:
        """
        Extract the 32-dimensional intermediate feature embedding.

        Returns
        -------
        np.ndarray of shape (32,), dtype float32
        """
        x = preprocess_image(image, device=self.device)
        with torch.no_grad():
            embedding = self.model.extract_embedding(x)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def predict_probabilities(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Predict sigmoid probabilities for all 4 urban labels.
        """
        x = preprocess_image(image, device=self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(x).squeeze(0).cpu().numpy()

        return {
            LABEL_NAMES[i]: round(float(probs[i]), 4)
            for i in range(NUM_LABELS)
        }
