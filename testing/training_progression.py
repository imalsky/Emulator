#!/usr/bin/env python3
"""Plot training progression from CSV log."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd


MODEL_NAME = "trained_model"  # folder under ./models/


def _get_col(df, *candidates, required=False):
    """Return the first existing column among candidates, else None (or raise)."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the required columns found: {candidates}")
    return None


def plot_training_progression(model_dir: Path):
    """Plot train/val loss curves and LR (if present)."""
    log_path = model_dir / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    df = pd.read_csv(log_path)

    # Column names can vary; be forgiving.
    epoch_col = _get_col(df, "epoch", "Epoch", required=True)
    train_col = _get_col(df, "train_loss", "train", "loss_train", required=True)
    val_col = _get_col(df, "val_loss", "valid_loss", "validation_loss")
    lr_col = _get_col(df, "lr", "learning_rate", "train_lr")

    has_val = val_col is not None
    has_lr = lr_col is not None

    # Determine subplot layout
    ncols = 2 if has_lr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12 if has_lr else 7, 5))
    if ncols == 1:
        axes = [axes]  # make indexable

    # Loss curves
    ax = axes[0]
    ax.plot(df[epoch_col], df[train_col], "o-", label="Train", alpha=0.8)
    if has_val:
        ax.plot(df[epoch_col], df[val_col], "s-", label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("Training Progression")
    ax.legend()

    # Learning rate (optional)
    if has_lr:
        ax2 = axes[1]
        ax2.plot(df[epoch_col], df[lr_col], ".-")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Learning Rate Schedule")

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "training_progression.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"✓ Saved training progression plot -> {out_path}")


if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent / "models" / MODEL_NAME
    plot_training_progression(model_dir)
