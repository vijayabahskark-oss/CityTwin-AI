"""
perception/train_cnn.py — Train Urban Perception CNN
====================================================
Trains UrbanPerceptionCNN for multi-label classification on synthetic
urban scenes, evaluates per-label precision/recall/F1 and exact match accuracy,
and saves model checkpoints and visualization plots.

Usage
-----
    python -m perception.train_cnn
    python -m perception.train_cnn --epochs 25 --batch-size 32 --seed 42
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
from torch.utils.data import DataLoader

from perception.synthetic_data import LABEL_NAMES, NUM_LABELS
from perception.dataset import build_perception_datasets, make_dataloader, UrbanPerceptionDataset
from perception.cnn_model import UrbanPerceptionCNN

OUTPUT_DIR = _ROOT / "outputs" / "perception"
MODEL_DIR = _ROOT / "models"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _evaluate_metrics(
    model: UrbanPerceptionCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """
    Compute multi-label classification metrics across test set.

    Returns
    -------
    (mean_loss, metrics_dict, confusion_matrices)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())

    n_samples = sum(p.shape[0] for p in all_preds)
    mean_loss = total_loss / max(n_samples, 1)

    preds_arr = np.concatenate(all_preds, axis=0)      # (N, 4)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 4)
    bin_preds = (preds_arr >= threshold).astype(int)
    bin_targets = (targets_arr >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    f1_list: List[float] = []
    conf_matrices = np.zeros((NUM_LABELS, 2, 2), dtype=int)  # [ [TN, FP], [FN, TP] ]

    for i, name in enumerate(LABEL_NAMES):
        p = bin_preds[:, i]
        t = bin_targets[:, i]

        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        tn = int(np.sum((p == 0) & (t == 0)))

        conf_matrices[i] = np.array([[tn, fp], [fn, tp]])

        acc = (tp + tn) / max(len(t), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0

        metrics[f"{name}_accuracy"] = float(acc)
        metrics[f"{name}_precision"] = float(prec)
        metrics[f"{name}_recall"] = float(rec)
        metrics[f"{name}_f1"] = float(f1)
        f1_list.append(f1)

    metrics["macro_f1"] = float(np.mean(f1_list))

    # Exact match ratio: all 4 labels must match exactly
    exact_matches = np.all(bin_preds == bin_targets, axis=1)
    metrics["exact_match_accuracy"] = float(np.mean(exact_matches))

    return mean_loss, metrics, conf_matrices


def _save_plots(
    train_losses: List[float],
    val_losses: List[float],
    conf_matrices: np.ndarray,
    output_dir: Path,
) -> None:
    """Save training loss curve and 4-panel confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] matplotlib not installed.")
        return

    # 1. Training Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train BCE Loss", color="#1E88E5", linewidth=2.0)
    ax.plot(epochs, val_losses, label="Val BCE Loss", color="#E53935", linestyle="--", linewidth=2.0)
    ax.set_title("Unit II — UrbanPerceptionCNN Training Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogits Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = output_dir / "training_curve.png"
    plt.savefig(str(curve_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curve saved to: {curve_path}")

    # 2. 4-Panel Confusion Matrix
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for i, name in enumerate(LABEL_NAMES):
        ax = axes_flat[i]
        cm = conf_matrices[i]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        clean_name = name.replace("_", " ").title()
        ax.set_title(f"{clean_name}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg (0)", "Pred Pos (1)"])
        ax.set_yticklabels(["True Neg (0)", "True Pos (1)"])

        # Add text annotations
        for y in range(2):
            for x in range(2):
                val = cm[y, x]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(x, y, f"{val}", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to: {cm_path}")


def train_cnn(
    num_samples: int = 1200,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.2,
    seed: int = 42,
    patience: int = 6,
    save_dir: Path = MODEL_DIR,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> Dict:
    """
    Train UrbanPerceptionCNN.
    """
    _set_seed(seed)
    device = torch.device("cpu")
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 65)
        print("  CityTwin AI — Training Urban Perception CNN (Stage 5 / Unit II)")
        print("=" * 65)
        print(f"[1/5] Generating {num_samples} synthetic urban scenes...")
        t0 = time.time()

    datasets = build_perception_datasets(num_samples=num_samples, seed=seed)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    if verbose:
        print(f"      train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}  [{time.time()-t0:.1f}s]")

    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    model = UrbanPerceptionCNN(in_channels=3, num_classes=NUM_LABELS, dropout=dropout).to(device)

    if verbose:
        print(f"[2/5] Architecture: {model}")

    criterion = nn.BCEWithLogitsLoss()
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict = {}
    history = {"train_loss": [], "val_loss": []}

    if verbose:
        print(f"[3/5] Training for up to {epochs} epochs (patience={patience})...")
        print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Macro F1':>9}  {'Exact Match':>11}")
        print("  " + "-" * 56)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_loss, val_metrics, _ = _evaluate_metrics(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}"
                f"  {val_metrics['macro_f1']:>9.3f}  {val_metrics['exact_match_accuracy']*100:>10.1f}%"
            )

        if val_loss < best_val_loss - 1e-4:
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

    test_loss, test_metrics, conf_matrices = _evaluate_metrics(model, test_loader, criterion, device)

    if verbose:
        print(f"  Test Loss (BCE)       : {test_loss:.4f}")
        print(f"  Macro F1 Score        : {test_metrics['macro_f1']:.4f}")
        print(f"  Exact Match Accuracy  : {test_metrics['exact_match_accuracy']*100:.2f}%\n")
        print(f"  {'Label':<22} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>6}")
        print("  " + "-" * 62)
        for name in LABEL_NAMES:
            acc = test_metrics[f"{name}_accuracy"] * 100
            prec = test_metrics[f"{name}_precision"]
            rec = test_metrics[f"{name}_recall"]
            f1 = test_metrics[f"{name}_f1"]
            clean_name = name.replace("_", " ").title()
            print(f"  {clean_name:<22} | {acc:>7.1f}% | {prec:>9.3f} | {rec:>8.3f} | {f1:>6.3f}")

    if verbose:
        print(f"\n[5/5] Saving model artifacts...")

    model_path = save_dir / "urban_cnn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "in_channels": 3,
                "num_classes": NUM_LABELS,
                "dropout": dropout,
            },
            "train_metrics": history,
            "test_metrics": test_metrics,
            "best_val_loss": best_val_loss,
            "epochs_trained": epochs_trained,
            "seed": seed,
        },
        str(model_path),
    )
    if verbose:
        print(f"  Model saved to: {model_path}")

    # Metrics JSON
    all_metrics = {
        "test_loss": test_loss,
        "macro_f1": test_metrics["macro_f1"],
        "exact_match_accuracy": test_metrics["exact_match_accuracy"],
        "per_label": {
            name: {
                "accuracy": test_metrics[f"{name}_accuracy"],
                "precision": test_metrics[f"{name}_precision"],
                "recall": test_metrics[f"{name}_recall"],
                "f1": test_metrics[f"{name}_f1"],
            }
            for name in LABEL_NAMES
        },
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    if verbose:
        print(f"  Metrics saved to: {metrics_path}")

    _save_plots(history["train_loss"], history["val_loss"], conf_matrices, output_dir)

    return {
        "model_path": str(model_path),
        "metrics": all_metrics,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CityTwin AI Urban Perception CNN")
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_cnn(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        verbose=not args.quiet,
    )
