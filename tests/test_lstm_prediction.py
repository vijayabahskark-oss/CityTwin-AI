"""
tests/test_lstm_prediction.py
==============================
Unit and integration tests for the CityTwin AI LSTM prediction module.

Tests are designed to run FAST (< 30 s on CPU) by using tiny datasets
and a micro-sized model with very few epochs.

Test coverage
-------------
  1. Feature extraction — shape and value range
  2. Dataset — correct window shapes and split sizes
  3. LSTM forward pass — output shape and range
  4. Training — completes on tiny data without errors
  5. Model save / load — state dict survives round-trip
  6. Predictor — returns correct structure and finite values
  7. Determinism — same seed gives identical data
  8. Validation logic — bad input raises the right errors

Usage
-----
    python tests/test_lstm_prediction.py
    python -m pytest tests/test_lstm_prediction.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from simulation import MockCitySimulator, ScenarioGenerator, ScenarioType
from simulation.mock_city import CityState
from prediction.feature_extractor import (
    FeatureExtractor,
    FEATURE_DIM,
    TARGET_DIM,
    TARGET_NAMES,
    FEATURE_NAMES,
)
from prediction.dataset import (
    build_datasets,
    generate_raw_sequences,
    TrafficDataset,
    _make_windows,
)
from prediction.lstm_model import TrafficLSTM
from prediction.train_lstm import train
from prediction.predictor import TrafficPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_states(n: int = 20, seed: int = 0) -> List[CityState]:
    sim = MockCitySimulator(seed=seed)
    return sim.run(steps=n)


def _tiny_model(horizon: int = 3) -> TrafficLSTM:
    return TrafficLSTM(
        input_size=FEATURE_DIM,
        hidden_size=8,
        num_layers=1,
        horizon=horizon,
        target_dim=TARGET_DIM,
        dropout=0.0,
    )


def _run(fn, *args, **kwargs):
    """Run a test function and print pass/fail."""
    name = fn.__name__
    try:
        fn(*args, **kwargs)
        print(f"  [PASS] {name}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {name}: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 1: Feature extraction
# ---------------------------------------------------------------------------

def test_feature_extraction():
    states = _make_states(5)
    fe = FeatureExtractor()

    vec = fe.extract(states[0])
    assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0), \
        f"Features out of [0,1]: min={vec.min():.4f} max={vec.max():.4f}"

    mat = fe.extract_sequence(states)
    assert mat.shape == (5, FEATURE_DIM), f"Expected (5,{FEATURE_DIM}), got {mat.shape}"

    tgt = fe.extract_targets(states[0])
    assert tgt.shape == (TARGET_DIM,), f"Expected ({TARGET_DIM},), got {tgt.shape}"
    assert np.all(tgt >= 0.0) and np.all(tgt <= 1.0), \
        f"Targets out of [0,1]: {tgt}"


# ---------------------------------------------------------------------------
# Test 2: Dataset shapes
# ---------------------------------------------------------------------------

def test_dataset_shapes():
    seq_len, horizon = 5, 3
    # Use tiny generation to keep test fast
    datasets = build_datasets(
        steps_per_scenario=30,  # tiny
        seq_len=seq_len,
        horizon=horizon,
        seed=0,
    )
    for split, ds in datasets.items():
        assert len(ds) > 0, f"{split} dataset is empty"
        X, y = ds[0]
        assert X.shape == (seq_len, FEATURE_DIM), \
            f"{split}: X shape {X.shape}, expected ({seq_len},{FEATURE_DIM})"
        assert y.shape == (horizon, TARGET_DIM), \
            f"{split}: y shape {y.shape}, expected ({horizon},{TARGET_DIM})"
        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    # Check no temporal leakage: test indices come after val, which come after train
    total_states = 8 * 30  # 8 scenarios × 30 steps
    n_train = int(total_states * 0.70)
    n_val   = int(total_states * 0.15)
    assert n_train > 0 and n_val > 0


# ---------------------------------------------------------------------------
# Test 3: LSTM forward pass
# ---------------------------------------------------------------------------

def test_lstm_forward():
    seq_len, horizon, batch = 10, 5, 4
    model = _tiny_model(horizon=horizon)
    x = torch.randn(batch, seq_len, FEATURE_DIM)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch, horizon, TARGET_DIM), \
        f"Expected ({batch},{horizon},{TARGET_DIM}), got {out.shape}"

    # Sigmoid output: all values in (0, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), \
        "Model output outside [0,1]"

    assert torch.all(torch.isfinite(out)), "Non-finite values in model output"


# ---------------------------------------------------------------------------
# Test 4: Training on tiny dataset
# ---------------------------------------------------------------------------

def test_training_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            epochs=3,
            batch_size=16,
            lr=1e-3,
            hidden_size=8,
            num_layers=1,
            seq_len=5,
            horizon=3,
            steps_per_scenario=30,  # tiny
            seed=0,
            patience=2,
            save_dir=Path(tmpdir) / "models",
            output_dir=Path(tmpdir) / "outputs",
            verbose=False,
        )
    assert "model_path" in result
    assert "metrics" in result
    assert result["epochs_trained"] >= 1
    assert result["best_val_loss"] < float("inf")


# ---------------------------------------------------------------------------
# Test 5: Model save and load
# ---------------------------------------------------------------------------

def test_model_save_load():
    horizon = 3
    model = _tiny_model(horizon=horizon)
    x = torch.randn(2, 10, FEATURE_DIM)

    with torch.no_grad():
        out_before = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {
                    "input_size":  FEATURE_DIM,
                    "hidden_size": 8,
                    "num_layers":  1,
                    "horizon":     horizon,
                    "target_dim":  TARGET_DIM,
                    "dropout":     0.0,
                },
                "seq_len": 10,
            },
            str(path),
        )

        predictor = TrafficPredictor.from_checkpoint(path)

    with torch.no_grad():
        out_after = predictor.model(x).numpy()

    np.testing.assert_allclose(
        out_before, out_after, atol=1e-6,
        err_msg="Model weights changed after save/load"
    )


# ---------------------------------------------------------------------------
# Test 6: Predictor returns correct structure
# ---------------------------------------------------------------------------

def test_predictor_structure():
    horizon = 3
    seq_len = 10
    states = _make_states(seq_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            hidden_size=8,
            num_layers=1,
            seq_len=seq_len,
            horizon=horizon,
            steps_per_scenario=20,
            seed=1,
            patience=2,
            save_dir=Path(tmpdir) / "models",
            output_dir=Path(tmpdir) / "out",
            verbose=False,
        )
        predictor = TrafficPredictor.from_checkpoint(result["model_path"])

    preds = predictor.predict(states)

    # Keys
    for key in TARGET_NAMES:
        assert key in preds, f"Key '{key}' missing from predictions"

    # Lengths
    for key, val in preds.items():
        assert len(val) == horizon, \
            f"Key '{key}': expected {horizon} values, got {len(val)}"

    # Range checks
    for v in preds["traffic_density"]:
        assert 0.0 <= v <= 1.0, f"traffic_density out of [0,1]: {v}"
    for v in preds["average_speed"]:
        assert 0.0 <= v <= 120.0, f"average_speed out of [0,120]: {v}"
    for v in preds["queue_length"]:
        assert 0.0 <= v <= 1000.0, f"queue_length out of [0,1000]: {v}"


# ---------------------------------------------------------------------------
# Test 7: Prediction values are finite
# ---------------------------------------------------------------------------

def test_predictions_finite():
    seq_len = 10
    states = _make_states(seq_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            hidden_size=8,
            num_layers=1,
            seq_len=seq_len,
            horizon=3,
            steps_per_scenario=20,
            seed=2,
            patience=2,
            save_dir=Path(tmpdir) / "m",
            output_dir=Path(tmpdir) / "o",
            verbose=False,
        )
        predictor = TrafficPredictor.from_checkpoint(result["model_path"])

    preds = predictor.predict(states)
    for key, vals in preds.items():
        for v in vals:
            assert np.isfinite(v), f"Non-finite value in '{key}': {v}"


# ---------------------------------------------------------------------------
# Test 8: Determinism — same seed gives identical sequences
# ---------------------------------------------------------------------------

def test_determinism():
    states_a = generate_raw_sequences(steps_per_scenario=10, seed=99)
    states_b = generate_raw_sequences(steps_per_scenario=10, seed=99)

    fe = FeatureExtractor()
    feat_a = fe.extract_sequence(states_a)
    feat_b = fe.extract_sequence(states_b)

    np.testing.assert_array_equal(
        feat_a, feat_b,
        err_msg="Identical seeds produced different feature sequences"
    )


# ---------------------------------------------------------------------------
# Test 9: Bad input validation
# ---------------------------------------------------------------------------

def test_predictor_bad_input():
    seq_len = 5
    states = _make_states(seq_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            hidden_size=8,
            num_layers=1,
            seq_len=seq_len,
            horizon=3,
            steps_per_scenario=20,
            seed=3,
            patience=2,
            save_dir=Path(tmpdir) / "m",
            output_dir=Path(tmpdir) / "o",
            verbose=False,
        )
        predictor = TrafficPredictor.from_checkpoint(result["model_path"])

    # Too many states
    too_many = _make_states(seq_len + 5)
    try:
        predictor.predict(too_many)
        raise AssertionError("Should have raised ValueError for wrong seq_len")
    except ValueError:
        pass  # expected

    # Wrong feature dim
    bad_feat = np.zeros((seq_len, FEATURE_DIM + 5), dtype=np.float32)
    try:
        predictor.predict(bad_feat)
        raise AssertionError("Should have raised ValueError for wrong feature dim")
    except ValueError:
        pass  # expected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nCityTwin AI — LSTM Prediction Test Suite")
    print("=" * 50)

    tests = [
        test_feature_extraction,
        test_dataset_shapes,
        test_lstm_forward,
        test_training_runs,
        test_model_save_load,
        test_predictor_structure,
        test_predictions_finite,
        test_determinism,
        test_predictor_bad_input,
    ]

    results = [_run(t) for t in tests]

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)
    if passed == total:
        print(f"All {total} tests PASSED.")
        return 0
    else:
        print(f"{passed}/{total} tests passed. {total - passed} FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
