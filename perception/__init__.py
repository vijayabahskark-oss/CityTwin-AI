"""
CityTwin AI — Urban Visual Perception Package (Unit II: CNN Perception)
========================================================================
Implements a multi-label Convolutional Neural Network (UrbanPerceptionCNN)
for camera-based coarse visual scene understanding:
  - Vehicle presence
  - Pedestrian presence
  - Road obstruction
  - Low visibility / weather conditions

Public API
----------
  UrbanPerceptionCNN              — PyTorch CNN architecture
  UrbanPerceptionPredictor        — High-level image prediction API
  UrbanPerceptionFeatureExtractor  — Intermediate 32-dim feature extractor
  perception_to_observables       — Maps probabilities to normalized indicators
  preprocess_image                — Image-to-tensor preprocessing helper
  UrbanPerceptionDataset          — PyTorch multi-label dataset
  build_perception_datasets       — 70/15/15 dataset builder
  generate_single_scene           — Procedural synthetic image renderer
  generate_synthetic_dataset      — Procedural dataset generator
  LABEL_NAMES                     — List of the 4 multi-label target names
  NUM_LABELS                      — Constant (4)
  IMAGE_SIZE                      — Constant (128, 128)
"""

from .synthetic_data import (
    LABEL_NAMES,
    NUM_LABELS,
    IMAGE_SIZE,
    generate_single_scene,
    generate_synthetic_dataset,
)
from .dataset import (
    UrbanPerceptionDataset,
    build_perception_datasets,
    make_dataloader,
)
from .cnn_model import UrbanPerceptionCNN
from .feature_extractor import (
    preprocess_image,
    perception_to_observables,
    UrbanPerceptionFeatureExtractor,
)
from .predictor import UrbanPerceptionPredictor

__all__ = [
    "UrbanPerceptionCNN",
    "UrbanPerceptionPredictor",
    "UrbanPerceptionFeatureExtractor",
    "perception_to_observables",
    "preprocess_image",
    "UrbanPerceptionDataset",
    "build_perception_datasets",
    "make_dataloader",
    "generate_single_scene",
    "generate_synthetic_dataset",
    "LABEL_NAMES",
    "NUM_LABELS",
    "IMAGE_SIZE",
]
