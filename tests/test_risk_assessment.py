"""
tests/test_risk_assessment.py
==============================
Unit and integration test suite for CityTwin AI Stage 3:
Feedforward Neural Network Risk Assessment (Unit I).

Test Suite
----------
  1. test_feature_builder_shape
  2. test_feature_values_finite
  3. test_risk_target_generation
  4. test_mlp_forward_pass
  5. test_training_small_dataset
  6. test_model_save_load
  7. test_predictor_structure
  8. test_risk_score_bounds
  9. test_risk_level_categories
  10. test_determinism_fixed_seed
  11. test_end_to_end_integration (Simulator -> LSTM -> RiskMLP -> Output)

Usage
-----
    pytest tests/test_risk_assessment.py -v
    python tests/test_risk_assessment.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pytest

from simulation import MockCitySimulator, ScenarioGenerator, ScenarioType
from simulation.mock_city import CityState
from prediction.predictor import TrafficPredictor
from risk.feature_builder import (
    RiskFeatureBuilder,
    generate_synthetic_risk_target,
    classify_risk_level,
    RiskLevel,
    RISK_FEATURE_DIM,
    RISK_FEATURE_NAMES,
)
from risk.risk_model import RiskMLP
from risk.train_risk import train, build_risk_training_data
from risk.predictor import RiskPredictor


# ---------------------------------------------------------------------------
# Test 1: Feature builder output shape
# ---------------------------------------------------------------------------

def test_feature_builder_shape():
    sim = MockCitySimulator(seed=42)
    state = sim.step()
    future_pred = {
        "traffic_density": [0.3, 0.35, 0.4, 0.42, 0.45],
        "average_speed": [45.0, 42.0, 40.0, 38.0, 35.0],
        "queue_length": [50.0, 70.0, 90.0, 110.0, 130.0],
    }
    builder = RiskFeatureBuilder()
    vec = builder.build_features(state, future_pred)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (RISK_FEATURE_DIM,), f"Expected ({RISK_FEATURE_DIM},), got {vec.shape}"
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 2: Feature values are finite
# ---------------------------------------------------------------------------

def test_feature_values_finite():
    sim = MockCitySimulator(seed=10)
    builder = RiskFeatureBuilder()

    for _ in range(10):
        state = sim.step()
        future_pred = {
            "traffic_density": [state.traffic_density] * 5,
            "average_speed": [state.average_speed] * 5,
            "queue_length": [state.queue_length] * 5,
        }
        vec = builder.build_features(state, future_pred)
        assert np.all(np.isfinite(vec)), f"Found non-finite values in features: {vec}"


# ---------------------------------------------------------------------------
# Test 3: Risk target generation works
# ---------------------------------------------------------------------------

def test_risk_target_generation():
    sim = MockCitySimulator(seed=0)
    normal_state = sim.step()
    future_pred = {
        "traffic_density": [0.25] * 5,
        "average_speed": [50.0] * 5,
        "queue_length": [50.0] * 5,
    }

    # Normal state risk
    norm_score = generate_synthetic_risk_target(normal_state, future_pred)
    assert 0.0 <= norm_score <= 100.0

    # Accident state risk escalation
    accident_state = sim.step()
    accident_state.accident = True
    accident_state.road_blockage = True
    accident_state.emergency_vehicle_count = 3
    accident_pred = {
        "traffic_density": [0.85] * 5,
        "average_speed": [5.0] * 5,
        "queue_length": [800.0] * 5,
    }
    acc_score = generate_synthetic_risk_target(accident_state, accident_pred)
    assert 0.0 <= acc_score <= 100.0
    assert acc_score > norm_score
    assert acc_score >= 75.0  # Accident triggers CRITICAL range


# ---------------------------------------------------------------------------
# Test 4: MLP forward pass works
# ---------------------------------------------------------------------------

def test_mlp_forward_pass():
    model = RiskMLP(input_dim=RISK_FEATURE_DIM, hidden_dims=[64, 32], dropout=0.0)
    model.eval()

    batch_size = 4
    x = torch.rand(batch_size, RISK_FEATURE_DIM)
    with torch.no_grad():
        out_norm = model(x)
        out_score = model.predict_score(x)

    assert out_norm.shape == (batch_size, 1)
    assert out_score.shape == (batch_size, 1)
    assert torch.all(out_norm >= 0.0) and torch.all(out_norm <= 1.0)
    assert torch.all(out_score >= 0.0) and torch.all(out_score <= 100.0)


# ---------------------------------------------------------------------------
# Test 5: Training completes on a small dataset
# ---------------------------------------------------------------------------

def test_training_small_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = train(
            epochs=3,
            batch_size=16,
            steps_per_scenario=20,
            seed=42,
            patience=2,
            save_dir=Path(tmpdir) / "models",
            output_dir=Path(tmpdir) / "outputs",
            verbose=False,
        )
        assert "model_path" in res
        assert Path(res["model_path"]).exists()
        assert res["best_val_loss"] < float("inf")
        assert "metrics" in res
        assert "mae" in res["metrics"]
        assert "classification_accuracy_pct" in res["metrics"]


# ---------------------------------------------------------------------------
# Test 6: Model save/load works
# ---------------------------------------------------------------------------

def test_model_save_load():
    model = RiskMLP(input_dim=RISK_FEATURE_DIM, hidden_dims=[32, 16], dropout=0.0)
    model.eval()
    x = torch.rand(2, RISK_FEATURE_DIM)

    with torch.no_grad():
        out1 = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_risk.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": RISK_FEATURE_DIM,
                    "hidden_dims": [32, 16],
                    "dropout": 0.0,
                },
            },
            str(path),
        )

        loaded_model = RiskMLP(input_dim=RISK_FEATURE_DIM, hidden_dims=[32, 16], dropout=0.0)
        ckpt = torch.load(str(path), weights_only=False)
        loaded_model.load_state_dict(ckpt["state_dict"])
        loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(x).numpy()

    np.testing.assert_allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Predictor returns correct structure
# ---------------------------------------------------------------------------

def test_predictor_structure():
    sim = MockCitySimulator(seed=42)
    state = sim.step()
    future_pred = {
        "traffic_density": [0.3] * 5,
        "average_speed": [45.0] * 5,
        "queue_length": [50.0] * 5,
    }

    model = RiskMLP(input_dim=RISK_FEATURE_DIM, hidden_dims=[32, 16])
    predictor = RiskPredictor(model=model)

    res = predictor.predict(state, future_pred)
    assert isinstance(res, dict)
    assert "risk_score" in res
    assert "risk_level" in res
    assert isinstance(res["risk_score"], (int, float))
    assert isinstance(res["risk_level"], str)


# ---------------------------------------------------------------------------
# Test 8: Risk score is between 0 and 100
# ---------------------------------------------------------------------------

def test_risk_score_bounds():
    sim = MockCitySimulator(seed=123)
    model = RiskMLP(input_dim=RISK_FEATURE_DIM)
    predictor = RiskPredictor(model=model)

    for _ in range(10):
        state = sim.step()
        future_pred = {
            "traffic_density": [np.random.uniform(0.0, 1.0) for _ in range(5)],
            "average_speed": [np.random.uniform(0.0, 120.0) for _ in range(5)],
            "queue_length": [np.random.uniform(0.0, 1000.0) for _ in range(5)],
        }
        res = predictor.predict(state, future_pred)
        assert 0.0 <= res["risk_score"] <= 100.0


# ---------------------------------------------------------------------------
# Test 9: Risk level categories are valid
# ---------------------------------------------------------------------------

def test_risk_level_categories():
    assert classify_risk_level(10.0) == RiskLevel.LOW.value
    assert classify_risk_level(30.0) == RiskLevel.MEDIUM.value
    assert classify_risk_level(60.0) == RiskLevel.HIGH.value
    assert classify_risk_level(85.0) == RiskLevel.CRITICAL.value

    valid_levels = {lvl.value for lvl in RiskLevel}
    for score in [0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 100.0]:
        assert classify_risk_level(score) in valid_levels


# ---------------------------------------------------------------------------
# Test 10: Deterministic behavior with fixed seed
# ---------------------------------------------------------------------------

def test_determinism_fixed_seed():
    sim1 = MockCitySimulator(seed=77)
    sim2 = MockCitySimulator(seed=77)

    s1 = sim1.step()
    s2 = sim2.step()

    pred = {"traffic_density": [0.4] * 5, "average_speed": [35.0] * 5, "queue_length": [100.0] * 5}

    builder = RiskFeatureBuilder()
    feat1 = builder.build_features(s1, pred)
    feat2 = builder.build_features(s2, pred)

    np.testing.assert_array_equal(feat1, feat2)

    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    target1 = generate_synthetic_risk_target(s1, pred, noise_std=2.0, rng=rng1)
    target2 = generate_synthetic_risk_target(s2, pred, noise_std=2.0, rng=rng2)

    assert target1 == target2


# ---------------------------------------------------------------------------
# Test 11: End-to-End Integration Test
# (MockCitySimulator -> LSTM Predictor -> FeatureBuilder -> RiskMLP)
# ---------------------------------------------------------------------------

def test_end_to_end_integration():
    # 1. Generate sequence from simulator
    sim = MockCitySimulator(seed=42)
    states = sim.run(steps=10)
    current_state = states[-1]

    # 2. LSTM Predictor (from saved checkpoint)
    lstm_ckpt = _ROOT / "models" / "best_lstm_model.pt"
    if not lstm_ckpt.exists():
        pytest.skip("LSTM model checkpoint required for integration test.")

    lstm_predictor = TrafficPredictor.from_checkpoint(lstm_ckpt)
    future_pred = lstm_predictor.predict(states)

    assert "traffic_density" in future_pred
    assert "average_speed" in future_pred
    assert "queue_length" in future_pred
    assert len(future_pred["traffic_density"]) == 5

    # 3. Feature Builder
    builder = RiskFeatureBuilder()
    feat_vec = builder.build_features(current_state, future_pred)
    assert feat_vec.shape == (RISK_FEATURE_DIM,)

    # 4. Risk Predictor
    risk_model = RiskMLP(input_dim=RISK_FEATURE_DIM, hidden_dims=[64, 32], dropout=0.0)
    risk_predictor = RiskPredictor(model=risk_model)

    result = risk_predictor.predict(current_state, future_pred)
    assert "risk_score" in result
    assert "risk_level" in result
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
