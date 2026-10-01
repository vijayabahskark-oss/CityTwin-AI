"""
dataset.py — RiskDataset & Data Generation
===========================================
PyTorch Dataset combining current physical state observations with
future temporal forecasts to train the RiskFNN model.

Input Feature Vector Representation (29 Floats)
-----------------------------------------------
  Indices 0..13 (14 features)  : Current CityState features (from FeatureExtractor)
  Indices 14..18 (5 features)  : Forecasted traffic density over 5 timesteps [0, 1]
  Indices 19..23 (5 features)  : Forecasted average speed normalized (/ 120)
  Indices 24..28 (5 features)  : Forecasted queue length normalized (/ 1000)

Target Vector Representation (4 Floats)
---------------------------------------
  Index 0 : Overall Urban Risk Score [0, 1]
  Index 1 : Congestion Risk [0, 1]
  Index 2 : Safety Hazard Risk [0, 1]
  Index 3 : Emergency Delay Risk [0, 1]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.mock_city import CityState
from prediction.feature_extractor import FeatureExtractor, FEATURE_DIM
from prediction.dataset import generate_raw_sequences, DEFAULT_SEQ_LEN, DEFAULT_HORIZON
from risk.risk_calculator import (
    compute_ground_truth_risk,
    RISK_TARGET_NAMES,
    RISK_TARGET_DIM,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_FEAT_DIM: int = FEATURE_DIM          # 14
FORECAST_DIM: int = DEFAULT_HORIZON * 3      # 5 * 3 = 15
RISK_FEATURE_DIM: int = CURRENT_FEAT_DIM + FORECAST_DIM  # 29


# ---------------------------------------------------------------------------
# Feature Vector Construction
# ---------------------------------------------------------------------------

def construct_risk_features(
    current_state: CityState,
    forecast: Dict[str, Sequence[float]],
    feature_extractor: Optional[FeatureExtractor] = None,
) -> np.ndarray:
    """
    Construct a 29-dimensional normalized float32 vector fusing current
    state observables and the 5-step forecast.

    Parameters
    ----------
    current_state : CityState
    forecast : dict with 'traffic_density', 'average_speed', 'queue_length'
    feature_extractor : FeatureExtractor | None

    Returns
    -------
    np.ndarray, shape (29,), dtype float32
    """
    fe = feature_extractor or FeatureExtractor()
    curr_vec = fe.extract(current_state)  # (14,)

    # Extract & normalize forecast components
    densities = np.array(forecast.get("traffic_density", [current_state.traffic_density] * 5), dtype=np.float32)
    speeds = np.array(forecast.get("average_speed", [current_state.average_speed] * 5), dtype=np.float32) / 120.0
    queues = np.array(forecast.get("queue_length", [current_state.queue_length] * 5), dtype=np.float32) / 1000.0

    # Ensure length 5
    if len(densities) < 5:
        densities = np.pad(densities, (0, 5 - len(densities)), mode="edge")
    if len(speeds) < 5:
        speeds = np.pad(speeds, (0, 5 - len(speeds)), mode="edge")
    if len(queues) < 5:
        queues = np.pad(queues, (0, 5 - len(queues)), mode="edge")

    densities = densities[:5]
    speeds = speeds[:5]
    queues = queues[:5]

    risk_vec = np.concatenate([curr_vec, densities, speeds, queues], axis=0).astype(np.float32)
    assert risk_vec.shape == (RISK_FEATURE_DIM,), f"Expected ({RISK_FEATURE_DIM},), got {risk_vec.shape}"
    return risk_vec


# ---------------------------------------------------------------------------
# Risk Dataset
# ---------------------------------------------------------------------------

class RiskDataset(Dataset):
    """
    PyTorch Dataset containing paired risk feature vectors (X) and
    ground-truth multi-target risk labels (y).

    Parameters
    ----------
    X : np.ndarray, shape (N, 29)
    y : np.ndarray, shape (N, 4)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def feature_dim(self) -> int:
        return self.X.shape[1]

    @property
    def target_dim(self) -> int:
        return self.y.shape[1]


# ---------------------------------------------------------------------------
# Dataset Builder with Temporal Split
# ---------------------------------------------------------------------------

def build_risk_datasets(
    steps_per_scenario: int = 300,
    seq_len: int = DEFAULT_SEQ_LEN,
    horizon: int = DEFAULT_HORIZON,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, RiskDataset]:
    """
    Generate simulation states across scenarios, compute paired risk features
    and ground truth targets, and perform a non-leaking temporal split.

    Returns
    -------
    dict with keys 'train', 'val', 'test' mapping to RiskDataset instances.
    """
    fe = FeatureExtractor()
    all_states = generate_raw_sequences(steps_per_scenario=steps_per_scenario, seed=seed)
    N = len(all_states)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # Slide through sequences
    for i in range(N - horizon):
        curr_state = all_states[i]
        future_states = all_states[i + 1 : i + 1 + horizon]

        # Construct forecast dictionary from future state window
        forecast = {
            "traffic_density": [s.traffic_density for s in future_states],
            "average_speed": [s.average_speed for s in future_states],
            "queue_length": [s.queue_length for s in future_states],
        }

        feat_vec = construct_risk_features(curr_state, forecast, fe)
        gt_risk = compute_ground_truth_risk(curr_state, forecast)

        target_vec = np.array([
            gt_risk["overall_risk"],
            gt_risk["congestion_risk"],
            gt_risk["safety_risk"],
            gt_risk["emergency_risk"],
        ], dtype=np.float32)

        X_list.append(feat_vec)
        y_list.append(target_vec)

    X_all = np.stack(X_list, axis=0)  # (M, 29)
    y_all = np.stack(y_list, axis=0)  # (M, 4)

    M = len(X_all)
    n_train = int(M * train_frac)
    n_val = int(M * val_frac)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    X_test, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]

    return {
        "train": RiskDataset(X_train, y_train),
        "val": RiskDataset(X_val, y_val),
        "test": RiskDataset(X_test, y_test),
    }


def make_risk_dataloader(
    dataset: RiskDataset,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience wrapper creating a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
    )
