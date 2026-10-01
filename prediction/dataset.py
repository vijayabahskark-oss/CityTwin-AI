"""
dataset.py — TrafficDataset & build_datasets
=============================================
Provides a PyTorch-compatible sliding-window dataset built from
MockCitySimulator sequences.

Sliding-window schema
---------------------
Given a raw time-series of N CityStates:

    [ s₀, s₁, …, s_{N-1} ]

Each sample is:
    X : states[i  : i+seq_len]          shape (seq_len, FEATURE_DIM)
    y : targets[i+seq_len : i+seq_len+horizon]  shape (horizon, TARGET_DIM)

The dataset is split 70 / 15 / 15 % (train / val / test) over the
concatenated sequence from all scenarios WITHOUT shuffling, to preserve
temporal ordering within each split.

Data generation
---------------
For each of the 8 ScenarioTypes, the simulator runs `steps_per_scenario`
steps.  All sequences are concatenated in a fixed order (controlled by
`seed`) then split.  This ensures:

  * No temporal leakage across splits (cut-points are in time order).
  * Enough diversity across scenario types.
  * Reproducibility when seed is fixed.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make prediction importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import MockCitySimulator, ScenarioGenerator, ScenarioType
from simulation.mock_city import CityState
from prediction.feature_extractor import FeatureExtractor, FEATURE_DIM, TARGET_DIM

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide a stub so the rest of the file parses without torch installed
    class Dataset:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEQ_LEN:  int = 10   # number of input timesteps
DEFAULT_HORIZON:  int = 5    # number of future steps to predict
DEFAULT_STEPS:    int = 300  # steps generated per scenario
DEFAULT_SEED:     int = 42


# ---------------------------------------------------------------------------
# Raw sequence generation
# ---------------------------------------------------------------------------

def generate_raw_sequences(
    steps_per_scenario: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> List[CityState]:
    """
    Generate one long time-series by running every scenario in order.

    Each scenario uses a fresh simulator so internal signal cycles restart,
    but the selection order and the seed are both fixed, making the output
    fully reproducible.

    Parameters
    ----------
    steps_per_scenario : int
        Number of timesteps generated per scenario (default 300).
    seed : int
        Master seed. Each scenario simulator gets seed + scenario_index.

    Returns
    -------
    list of CityState, length = 8 * steps_per_scenario
    """
    gen = ScenarioGenerator()
    all_states: List[CityState] = []

    scenarios = [
        ScenarioType.NORMAL,
        ScenarioType.RUSH_HOUR,
        ScenarioType.ACCIDENT,
        ScenarioType.HEAVY_RAIN,
        ScenarioType.HEAVY_RAIN_FOG,
        ScenarioType.ACCIDENT_RAIN,
        ScenarioType.ROAD_BLOCKAGE,
        ScenarioType.EMERGENCY,
    ]

    for idx, scenario in enumerate(scenarios):
        sim = MockCitySimulator(seed=seed + idx)
        overlay = gen.get_overlay(scenario)
        states = sim.run(steps=steps_per_scenario, overlay=overlay)
        all_states.extend(states)

    return all_states


# ---------------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------------

def _make_windows(
    features: np.ndarray,   # (N, FEATURE_DIM)
    targets: np.ndarray,    # (N, TARGET_DIM)
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert feature/target arrays into sliding-window pairs.

    Returns
    -------
    X : np.ndarray, shape (num_windows, seq_len, FEATURE_DIM)
    y : np.ndarray, shape (num_windows, horizon, TARGET_DIM)
    """
    N = len(features)
    num_windows = N - seq_len - horizon + 1
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data: N={N}, seq_len={seq_len}, horizon={horizon}. "
            f"Need at least {seq_len + horizon} steps."
        )

    X = np.stack(
        [features[i : i + seq_len] for i in range(num_windows)], axis=0
    ).astype(np.float32)

    y = np.stack(
        [targets[i + seq_len : i + seq_len + horizon] for i in range(num_windows)],
        axis=0,
    ).astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# TrafficDataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    PyTorch Dataset that wraps a pre-computed set of (X, y) windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, FEATURE_DIM)
    y : np.ndarray, shape (N, horizon, TARGET_DIM)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install it with: pip install torch")
        self.X = torch.from_numpy(X)   # (N, seq_len, FEATURE_DIM)
        self.y = torch.from_numpy(y)   # (N, horizon, TARGET_DIM)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        return self.X[idx], self.y[idx]

    @property
    def seq_len(self) -> int:
        return self.X.shape[1]

    @property
    def horizon(self) -> int:
        return self.y.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.X.shape[2]

    @property
    def target_dim(self) -> int:
        return self.y.shape[2]


# ---------------------------------------------------------------------------
# build_datasets — main entry point
# ---------------------------------------------------------------------------

def build_datasets(
    steps_per_scenario: int = DEFAULT_STEPS,
    seq_len: int = DEFAULT_SEQ_LEN,
    horizon: int = DEFAULT_HORIZON,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test_frac is the remainder
) -> Dict[str, "TrafficDataset"]:
    """
    Generate simulation data and return train / val / test datasets.

    The split is done on the raw time-series (no shuffling) to avoid
    temporal leakage:

        [ ─────── train (70%) ─────── | ── val (15%) ── | ── test (15%) ── ]

    Parameters
    ----------
    steps_per_scenario : int
        Simulation steps per scenario (default 300 → 2 400 total states).
    seq_len : int
        Input sequence length in timesteps (default 10).
    horizon : int
        Prediction horizon in timesteps (default 5).
    seed : int
        Master random seed.
    train_frac, val_frac : float
        Fractions of the raw series used for training and validation.

    Returns
    -------
    dict with keys "train", "val", "test" → TrafficDataset instances
    """
    fe = FeatureExtractor()

    # 1. Generate raw states
    all_states = generate_raw_sequences(
        steps_per_scenario=steps_per_scenario, seed=seed
    )
    N = len(all_states)

    # 2. Extract feature and target arrays
    features = fe.extract_sequence(all_states)       # (N, FEATURE_DIM)
    targets  = fe.extract_target_sequence(all_states) # (N, TARGET_DIM)

    # 3. Temporal split on raw series
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    # Ensure split boundaries respect the window size
    # (split points must leave at least seq_len+horizon on each side)
    min_split = seq_len + horizon
    n_train = max(n_train, min_split)
    n_val   = max(n_val,   min_split)

    feat_train = features[:n_train]
    tgt_train  = targets [:n_train]

    feat_val = features[n_train : n_train + n_val]
    tgt_val  = targets [n_train : n_train + n_val]

    feat_test = features[n_train + n_val :]
    tgt_test  = targets [n_train + n_val :]

    # 4. Build sliding-window pairs
    X_train, y_train = _make_windows(feat_train, tgt_train, seq_len, horizon)
    X_val,   y_val   = _make_windows(feat_val,   tgt_val,   seq_len, horizon)
    X_test,  y_test  = _make_windows(feat_test,  tgt_test,  seq_len, horizon)

    return {
        "train": TrafficDataset(X_train, y_train),
        "val":   TrafficDataset(X_val,   y_val),
        "test":  TrafficDataset(X_test,  y_test),
    }


def make_dataloader(
    dataset: "TrafficDataset",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> "DataLoader":
    """Convenience wrapper around torch DataLoader."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
