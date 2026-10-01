"""
perception/dataset.py — PyTorch Dataset for Urban Scene Multi-Label Perception
=============================================================================
Manages synthetic 128x128 RGB urban scene datasets, normalization transforms,
and 70/15/15 train/val/test splits for CNN multi-label training.

Labels (4 Dimensions)
---------------------
  0: vehicle_present     (0.0 or 1.0)
  1: pedestrian_present  (0.0 or 1.0)
  2: road_obstruction    (0.0 or 1.0)
  3: low_visibility      (0.0 or 1.0)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.synthetic_data import (
    generate_synthetic_dataset,
    LABEL_NAMES,
    NUM_LABELS,
    IMAGE_SIZE,
)


class UrbanPerceptionDataset(Dataset):
    """
    PyTorch Dataset for Urban Visual Perception.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, 128, 128, 3), dtype uint8 or float32.
    labels : np.ndarray
        Array of shape (N, 4), dtype float32.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        # Convert (N, H, W, C) uint8 -> (N, C, H, W) float32 in [0, 1]
        if images.dtype == np.uint8:
            imgs_f = images.astype(np.float32) / 255.0
        else:
            imgs_f = images.astype(np.float32)

        # Transpose HWC -> CHW
        if imgs_f.ndim == 4 and imgs_f.shape[-1] == 3:
            imgs_f = np.transpose(imgs_f, (0, 3, 1, 2))

        self.images = torch.from_numpy(imgs_f)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def build_perception_datasets(
    num_samples: int = 1200,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, UrbanPerceptionDataset]:
    """
    Generate synthetic data and return train, validation, and test datasets.
    """
    images, labels = generate_synthetic_dataset(num_samples=num_samples, seed=seed)

    N = len(images)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_imgs, train_lbls = images[:n_train], labels[:n_train]
    val_imgs, val_lbls = images[n_train : n_train + n_val], labels[n_train : n_train + n_val]
    test_imgs, test_lbls = images[n_train + n_val :], labels[n_train + n_val :]

    return {
        "train": UrbanPerceptionDataset(train_imgs, train_lbls),
        "val": UrbanPerceptionDataset(val_imgs, val_lbls),
        "test": UrbanPerceptionDataset(test_imgs, test_lbls),
    }


def make_dataloader(
    dataset: UrbanPerceptionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience wrapper creating a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
    )
