"""
CityTwin AI — Prediction Package (Unit III: LSTM Temporal Prediction)
======================================================================
Converts sequences of CityState observations into future traffic
forecasts using a lightweight LSTM model.

Pipeline position
-----------------
  MockCitySimulator  →  FeatureExtractor  →  TrafficDataset
       ↓                                           ↓
  CityState sequence              LSTM (train_lstm.py trains this)
                                             ↓
                                    TrafficPredictor (inference)
                                             ↓
                             {traffic_density, average_speed, queue_length}

Public API
----------
  FeatureExtractor   — CityState  →  normalised float32 numpy vector
  TrafficDataset     — sliding-window PyTorch Dataset
  TrafficLSTM        — PyTorch nn.Module
  TrafficPredictor   — load-and-infer wrapper (no PyTorch knowledge needed)
"""

from .feature_extractor import FeatureExtractor, FEATURE_NAMES, TARGET_NAMES
from .dataset import TrafficDataset, build_datasets
from .lstm_model import TrafficLSTM
from .predictor import TrafficPredictor

__all__ = [
    "FeatureExtractor",
    "FEATURE_NAMES",
    "TARGET_NAMES",
    "TrafficDataset",
    "build_datasets",
    "TrafficLSTM",
    "TrafficPredictor",
]
