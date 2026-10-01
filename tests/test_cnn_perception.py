"""
tests/test_cnn_perception.py
=============================
Comprehensive unit and integration test suite for Stage 5:
Urban Visual Perception Module (Unit II: CNN).

Test Coverage
-------------
  1. test_synthetic_image_generation
  2. test_synthetic_data_deterministic
  3. test_label_validity
  4. test_dataset_shapes_and_splits
  5. test_cnn_forward_pass
  6. test_cnn_output_probabilities
  7. test_feature_extraction
  8. test_model_save_load
  9. test_predictor_api
  10. test_perception_to_observables
  11. test_short_training_run
  12. test_deterministic_inference
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pytest
from PIL import Image

from perception.synthetic_data import (
    generate_single_scene,
    generate_synthetic_dataset,
    LABEL_NAMES,
    NUM_LABELS,
    IMAGE_SIZE,
)
from perception.dataset import (
    UrbanPerceptionDataset,
    build_perception_datasets,
    make_dataloader,
)
from perception.cnn_model import UrbanPerceptionCNN
from perception.feature_extractor import (
    preprocess_image,
    perception_to_observables,
    UrbanPerceptionFeatureExtractor,
)
from perception.predictor import UrbanPerceptionPredictor
from perception.train_cnn import train_cnn


# ---------------------------------------------------------------------------
# Test 1: Synthetic Image Generation
# ---------------------------------------------------------------------------

def test_synthetic_image_generation():
    img = generate_single_scene(has_vehicle=True, has_pedestrian=True, has_obstruction=False, has_low_visibility=False)
    assert isinstance(img, np.ndarray)
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


# ---------------------------------------------------------------------------
# Test 2: Deterministic Data Generation with Seed
# ---------------------------------------------------------------------------

def test_synthetic_data_deterministic():
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)

    img1 = generate_single_scene(True, False, True, False, rng=rng1)
    img2 = generate_single_scene(True, False, True, False, rng=rng2)

    np.testing.assert_array_equal(img1, img2)

    imgs1, lbls1 = generate_synthetic_dataset(num_samples=10, seed=123)
    imgs2, lbls2 = generate_synthetic_dataset(num_samples=10, seed=123)

    np.testing.assert_array_equal(imgs1, imgs2)
    np.testing.assert_array_equal(lbls1, lbls2)


# ---------------------------------------------------------------------------
# Test 3: Label Validity
# ---------------------------------------------------------------------------

def test_label_validity():
    _, labels = generate_synthetic_dataset(num_samples=30, seed=7)
    assert labels.shape == (30, NUM_LABELS)
    assert set(np.unique(labels)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Test 4: Dataset Shapes and Splits
# ---------------------------------------------------------------------------

def test_dataset_shapes_and_splits():
    datasets = build_perception_datasets(num_samples=40, seed=1)
    assert "train" in datasets and "val" in datasets and "test" in datasets

    for split, ds in datasets.items():
        assert len(ds) > 0
        img, lbl = ds[0]
        assert img.shape == (3, 128, 128)
        assert lbl.shape == (NUM_LABELS,)
        assert img.dtype == torch.float32
        assert lbl.dtype == torch.float32
        assert torch.all(img >= 0.0) and torch.all(img <= 1.0)


# ---------------------------------------------------------------------------
# Test 5: CNN Forward Pass
# ---------------------------------------------------------------------------

def test_cnn_forward_pass():
    model = UrbanPerceptionCNN(in_channels=3, num_classes=NUM_LABELS)
    model.eval()

    batch_size = 4
    x = torch.rand(batch_size, 3, 128, 128)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, NUM_LABELS)
    assert torch.all(torch.isfinite(logits))


# ---------------------------------------------------------------------------
# Test 6: Output Probabilities Range
# ---------------------------------------------------------------------------

def test_cnn_output_probabilities():
    model = UrbanPerceptionCNN()
    model.eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        probs = model.predict_proba(x)

    assert probs.shape == (2, NUM_LABELS)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert torch.all(torch.isfinite(probs))


# ---------------------------------------------------------------------------
# Test 7: Intermediate Feature Extraction (32-dim Embedding)
# ---------------------------------------------------------------------------

def test_feature_extraction():
    model = UrbanPerceptionCNN()
    extractor = UrbanPerceptionFeatureExtractor(model)

    raw_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    embedding = extractor.extract_features(raw_img)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (32,)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# Test 8: Model Save & Load Roundtrip
# ---------------------------------------------------------------------------

def test_model_save_load():
    model = UrbanPerceptionCNN(dropout=0.0)
    model.eval()
    x = torch.rand(2, 3, 128, 128)

    with torch.no_grad():
        out1 = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_cnn.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {"in_channels": 3, "num_classes": NUM_LABELS, "dropout": 0.0},
            },
            str(path),
        )

        loaded_model = UrbanPerceptionCNN(dropout=0.0)
        ckpt = torch.load(str(path), weights_only=False)
        loaded_model.load_state_dict(ckpt["state_dict"])
        loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(x).numpy()

    np.testing.assert_allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 9: Predictor API
# ---------------------------------------------------------------------------

def test_predictor_api():
    model = UrbanPerceptionCNN()
    predictor = UrbanPerceptionPredictor(model)

    raw_img = np.zeros((128, 128, 3), dtype=np.uint8)
    result = predictor.predict(raw_img)

    assert isinstance(result, dict)
    for name in LABEL_NAMES:
        assert name in result
        assert 0.0 <= result[name] <= 1.0


# ---------------------------------------------------------------------------
# Test 10: Perception to Observables Mapping
# ---------------------------------------------------------------------------

def test_perception_to_observables():
    sample_preds = {
        "vehicle_present": 0.85,
        "pedestrian_present": 0.40,
        "road_obstruction": 0.10,
        "low_visibility": 0.90,
    }
    observables = perception_to_observables(sample_preds)

    assert "visual_vehicle_indicator" in observables
    assert "visual_pedestrian_indicator" in observables
    assert "visual_obstruction_indicator" in observables
    assert "visual_visibility_indicator" in observables

    assert observables["visual_vehicle_indicator"] == 0.85
    assert observables["visual_visibility_indicator"] == 0.10  # 1.0 - 0.90


# ---------------------------------------------------------------------------
# Test 11: Short Training Run on Small Dataset
# ---------------------------------------------------------------------------

def test_short_training_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = train_cnn(
            num_samples=40,
            epochs=2,
            batch_size=16,
            seed=42,
            patience=2,
            save_dir=Path(tmpdir) / "models",
            output_dir=Path(tmpdir) / "outputs",
            verbose=False,
        )

        assert "model_path" in res
        assert Path(res["model_path"]).exists()
        assert "metrics" in res
        assert "macro_f1" in res["metrics"]
        assert "exact_match_accuracy" in res["metrics"]


# ---------------------------------------------------------------------------
# Test 12: Deterministic Inference
# ---------------------------------------------------------------------------

def test_deterministic_inference():
    model = UrbanPerceptionCNN()
    predictor = UrbanPerceptionPredictor(model)

    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    p1 = predictor.predict(img)
    p2 = predictor.predict(img)

    assert p1 == p2


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
