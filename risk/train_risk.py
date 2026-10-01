"""
risk/train_risk.py — Train Feedforward Risk Model
=================================================
Trains the RiskMLP model on data generated from MockCitySimulator +
TrafficPredictor, evaluates regression (MAE, RMSE) and classification
accuracy (LOW/MEDIUM/HIGH/CRITICAL), and saves model artifacts.

Usage
-----
    python risk/train_risk.py
    python risk/train_risk.py --epochs 40 --batch-size 64 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from simulation import MockCitySimulator, ScenarioGenerator, ScenarioType
from simulation.mock_city import CityState
from prediction.predictor import TrafficPredictor
from prediction.dataset import generate_raw_sequences, DEFAULT_SEQ_LEN, DEFAULT_HORIZON
from risk.feature_builder import (
    RiskFeatureBuilder,
    generate_synthetic_risk_target,
    classify_risk_level,
    RISK_FEATURE_DIM,
)
from risk.risk_model import RiskMLP

OUTPUT_DIR = _ROOT / "outputs" / "risk"
MODEL_DIR = _ROOT / "models"


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------

class RiskDataset(Dataset):
    """PyTorch Dataset for Risk Feature Vectors & Normalized Targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_risk_training_data(
    steps_per_scenario: int = 300,
    seq_len: int = DEFAULT_SEQ_LEN,
    horizon: int = DEFAULT_HORIZON,
    seed: int = 42,
    lstm_checkpoint: Path = MODEL_DIR / "best_lstm_model.pt",
    noise_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simulation states and use LSTM predictor to compute paired
    risk features and synthetic disruption targets.
    """
    builder = RiskFeatureBuilder()
    rng = np.random.RandomState(seed)

    # Load LSTM predictor if available
    lstm_predictor: Optional[TrafficPredictor] = None
    if lstm_checkpoint.exists():
        lstm_predictor = TrafficPredictor.from_checkpoint(lstm_checkpoint)

    all_states = generate_raw_sequences(steps_per_scenario=steps_per_scenario, seed=seed)
    N = len(all_states)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for i in range(seq_len - 1, N - horizon):
        history_window = all_states[i - (seq_len - 1) : i + 1]
        current_state = history_window[-1]

        # Obtain future forecast
        if lstm_predictor is not None:
            future_pred = lstm_predictor.predict(history_window)
        else:
            future_states = all_states[i + 1 : i + 1 + horizon]
            future_pred = {
                "traffic_density": [s.traffic_density for s in future_states],
                "average_speed": [s.average_speed for s in future_states],
                "queue_length": [s.queue_length for s in future_states],
            }

        # Build feature vector
        feat_vec = builder.build_features(current_state, future_pred)

        # Generate synthetic target (0-100)
        target_score = generate_synthetic_risk_target(
            current_state=current_state,
            future_prediction=future_pred,
            noise_std=noise_std,
            rng=rng,
        )

        X_list.append(feat_vec)
        y_list.append(target_score / 100.0)  # Normalize to [0, 1] for model

    X_all = np.stack(X_list, axis=0)
    y_all = np.array(y_list, dtype=np.float32)

    return X_all, y_all


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: RiskMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate regression loss, scaled MAE, scaled RMSE, and classification accuracy.

    Returns
    -------
    (mean_loss, mae_scaled, rmse_scaled, class_accuracy_pct)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(X)
            all_preds.extend(pred.squeeze(1).cpu().numpy().tolist())
            all_targets.extend(y.squeeze(1).cpu().numpy().tolist())

    n = len(all_preds)
    mean_loss = total_loss / max(n, 1)

    preds_100 = np.array(all_preds) * 100.0
    targets_100 = np.array(all_targets) * 100.0

    mae = float(np.mean(np.abs(preds_100 - targets_100)))
    rmse = float(np.sqrt(np.mean((preds_100 - targets_100) ** 2)))

    # Classification accuracy (LOW/MEDIUM/HIGH/CRITICAL)
    correct_classes = sum(
        1 for p, t in zip(preds_100, targets_100) if classify_risk_level(p) == classify_risk_level(t)
    )
    accuracy_pct = (correct_classes / max(n, 1)) * 100.0

    return mean_loss, mae, rmse, accuracy_pct


def _save_evaluation_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    """Plot actual vs predicted risk scores for test set."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] matplotlib not installed.")
        return

    n = min(150, len(actual))
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(actual[:n], label="Synthetic Target", color="#1E88E5", linewidth=1.5)
    ax.plot(predicted[:n], label="RiskMLP Predicted", color="#E53935", linestyle="--", linewidth=1.5, alpha=0.9)

    ax.axhline(25, color="#757575", linestyle=":", alpha=0.5, label="Low/Medium (25)")
    ax.axhline(50, color="#FF9800", linestyle=":", alpha=0.5, label="Medium/High (50)")
    ax.axhline(75, color="#F44336", linestyle=":", alpha=0.5, label="High/Critical (75)")

    ax.set_title("Unit I — RiskMLP: Actual vs Predicted Disruption Severity", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Risk Score (0–100)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 2e-3,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.15,
    steps_per_scenario: int = 300,
    seed: int = 42,
    patience: int = 8,
    save_dir: Path = MODEL_DIR,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> Dict:
    """
    Train the RiskMLP neural network.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    hidden_dims = hidden_dims or [64, 32]

    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[1/5] Generating risk training data (Simulation + LSTM)...")
        t0 = time.time()

    X_all, y_all = build_risk_training_data(
        steps_per_scenario=steps_per_scenario,
        seed=seed,
    )

    # 70 / 15 / 15 temporal split
    M = len(X_all)
    n_train = int(M * 0.70)
    n_val = int(M * 0.15)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    X_test, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]

    train_ds = RiskDataset(X_train, y_train)
    val_ds = RiskDataset(X_val, y_val)
    test_ds = RiskDataset(X_test, y_test)

    if verbose:
        print(f"      train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}  [{time.time()-t0:.1f}s]")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model instantiation
    model = RiskMLP(
        input_dim=RISK_FEATURE_DIM,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    if verbose:
        print(f"[2/5] Architecture: {model}")

    criterion = nn.SmoothL1Loss()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=4)

    if verbose:
        print(f"[3/5] Training for up to {epochs} epochs (patience={patience})...")
        print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val MAE':>8}  {'LR':>8}")
        print("  " + "-" * 48)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict = {}
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_loss, val_mae, val_rmse, val_acc = _evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        curr_lr = optimiser.param_groups[0]["lr"]

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  {epoch:>5}  {train_loss:>10.6f}  {val_loss:>10.6f}  {val_mae:>8.2f}  {curr_lr:>8.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n  [Early stopping triggered at epoch {epoch}]")
                break

    epochs_trained = epoch
    model.load_state_dict(best_state)

    if verbose:
        print(f"\n[4/5] Evaluating on test set...")

    test_loss, test_mae, test_rmse, test_acc = _evaluate_model(model, test_loader, criterion, device)

    if verbose:
        print(f"  Test SmoothL1 Loss : {test_loss:.6f}")
        print(f"  Test Score MAE     : {test_mae:.2f} / 100")
        print(f"  Test Score RMSE    : {test_rmse:.2f} / 100")
        print(f"  Level Accuracy     : {test_acc:.2f}% (LOW/MEDIUM/HIGH/CRITICAL)")

    if verbose:
        print(f"\n[5/5] Saving artifacts...")

    model_path = save_dir / "risk_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": RISK_FEATURE_DIM,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
            },
            "train_metrics": history,
            "test_metrics": {
                "test_loss": test_loss,
                "mae": test_mae,
                "rmse": test_rmse,
                "accuracy_pct": test_acc,
            },
            "best_val_loss": best_val_loss,
            "epochs_trained": epochs_trained,
            "seed": seed,
        },
        str(model_path),
    )
    if verbose:
        print(f"  Model saved to: {model_path}")

    # Metrics JSON
    metrics_info = {
        "test_loss": test_loss,
        "mae": test_mae,
        "rmse": test_rmse,
        "classification_accuracy_pct": test_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_info, f, indent=2)
    if verbose:
        print(f"  Metrics saved to: {metrics_path}")

    # Plot
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test).to(device)).squeeze(1).numpy() * 100.0
    test_actual = y_test * 100.0

    plot_path = output_dir / "actual_vs_predicted_risk.png"
    _save_evaluation_plot(test_actual, test_preds, plot_path)

    return {
        "model_path": str(model_path),
        "metrics": metrics_info,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CityTwin AI RiskMLP Model")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    res = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_scenario=args.steps,
        seed=args.seed,
        patience=args.patience,
        verbose=not args.quiet,
    )
    print(f"\nDone. Risk model saved to: {res['model_path']}")
