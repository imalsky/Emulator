# File 1: plot_training_progression.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Relative path to the model directory (assuming script is in 'testing/' and 'models/' is sibling)
model_dir = (
    Path(__file__).parent.parent / "models" / "trained_model_picaso_transformer_v2"
)  # Change 'trained_model_picaso_transformer' if needed


def plot_training_progress(log_path: Path, plots_dir: Path):
    # Load the training log CSV
    df = pd.read_csv(log_path)

    # Plot train and val loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Normalized MSE)")
    plt.title("Training Progression")
    plt.yscale("log")  # Log scale for better visibility of progression
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "training_progression.png")
    # plt.show()


if __name__ == "__main__":
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "training_log.csv"
    plot_training_progress(log_path, plots_dir)
